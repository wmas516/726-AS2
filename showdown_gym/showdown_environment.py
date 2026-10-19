import os
import time
from typing import Any, Dict
import numpy as np

from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player
from .base_environment import BaseShowdownEnv

from poke_env.data import GenData

# Load Gen9 data (type chart etc.)
GEN_DATA = GenData.from_gen(9)

_BATTLE_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_STATUS_EFFECTS = ("brn", "par", "psn", "tox", "slp", "frz")


def _boost_multiplier(stage: int, *, accuracy: bool = False) -> float:
    """Convert a Pokémon stat stage (-6..6) to its in-battle multiplier."""
    stage = max(-6, min(6, int(stage)))
    if accuracy:
        return (3 + stage) / 3 if stage >= 0 else 3 / (3 - stage)
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def _condition_name(value) -> str:
    """Normalize poke-env enums and strings for condition lookup."""
    return str(getattr(value, "name", value)).lower().replace("_", "").split(".")[-1]


def _status_embedding(pokemon) -> np.ndarray:
    """One-hot encode the active Pokémon's major status condition."""
    features = np.zeros(len(_STATUS_EFFECTS), dtype=np.float32)
    status = getattr(pokemon, "status", None)
    if status is None:
        return features
    name = _condition_name(status)
    aliases = {
        "burn": "brn", "burned": "brn",
        "paralysis": "par", "paralyzed": "par",
        "poison": "psn", "poisoned": "psn",
        "toxic": "tox", "badlypoisoned": "tox",
        "sleep": "slp", "asleep": "slp",
        "freeze": "frz", "frozen": "frz",
    }
    name = aliases.get(name, name)
    if name in _STATUS_EFFECTS:
        features[_STATUS_EFFECTS.index(name)] = 1.0
    return features


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.rl_agent = account_name_one
        self._prev_battle_state = {}
        # self._active_damage_multiplier = 1.0
        # self._opponent_damage_multiplier = 1.0

    # =========================================================
    # Action space
    # =========================================================
    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        # Limiting action space to only 5 switches and 4 moves
        return 9  # default 26-action mapping used by CARES

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 - 5 : switch
        6 - 9: move
        10 - 13: move and mega evolve
        14 - 17: move and z-move
        18 - 21: move and dynamax
        22 - 25: move and terastallize

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        # Limiting action space to only 5 switches and 4 moves
        if (action >=  10 or action > 0): 
            return (-2)
        return action

    def _move_damage_multiplier(self, battle: AbstractBattle, move, opponent) -> float:
        """
        Combine the move's type effectiveness with the relevant offensive and defensive stat boosts.
        """
        return self._calculate_damage_multiplier(
            move, battle.active_pokemon, opponent
        )

    def _calculate_damage_multiplier(self, move, attacker, defender) -> float:
        """Calculate a move's type effectiveness and stat-stage multiplier."""
        multiplier = 1.0

        try:
            if move.type and defender.type_1:
                multiplier = float(
                    move.type.damage_multiplier(
                        defender.type_1,
                        getattr(defender, "type_2", None),
                        type_chart=GEN_DATA.type_chart,
                    )
                )
        except Exception:
            multiplier = 1.0

        move_category = getattr(move.category, "name", str(move.category)).upper()
        multiplier *= self._stat_stage_damage_multiplier(
            move_category, attacker, defender
        )

        return float(multiplier)

    @staticmethod
    def _stage_multiplier(stage: int) -> float:
        """Convert a Pokémon stat stage into its battle multiplier."""
        if stage >= 0:
            return float((2 + stage) / 2)
        return float(2 / (2 - stage))

    def _stat_stage_damage_multiplier(self, category: str, attacker, defender) -> float:
        """Return the attacker/defender stat-stage ratio for a move category."""
        if category == "PHYSICAL":
            attack_stat, defense_stat = "atk", "def"
        elif category == "SPECIAL":
            attack_stat, defense_stat = "spa", "spd"
        else:
            return 1.0

        attack_stage = int(attacker.boosts.get(attack_stat, 0)) if attacker is not None else 0
        defense_stage = int(defender.boosts.get(defense_stat, 0)) if defender is not None else 0
        return self._stage_multiplier(attack_stage) / self._stage_multiplier(defense_stage)

    def _stat_effect_potential(self, battle: AbstractBattle | None) -> float:
        """Value active stat changes in the context of the current matchup."""
        if battle is None or not battle.active_pokemon or not battle.opponent_active_pokemon:
            return 0.0

        ally = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        ally_moves = list(getattr(battle, "available_moves", []) or [])
        opponent_moves = list((getattr(opponent, "moves", {}) or {}).values())

        def relevant_categories(moves):
            categories = {
                getattr(getattr(move, "category", None), "name", "").upper()
                for move in moves
                if getattr(move, "base_power", 0) > 0
            }
            categories.intersection_update({"PHYSICAL", "SPECIAL"})
            return categories or {"PHYSICAL", "SPECIAL"}

        ally_offense = max(
            self._stat_stage_damage_multiplier(category, ally, opponent)
            for category in relevant_categories(ally_moves)
        )
        opponent_offense = max(
            self._stat_stage_damage_multiplier(category, opponent, ally)
            for category in relevant_categories(opponent_moves)
        )

        ally_boosts = getattr(ally, "boosts", None) or {}
        opponent_boosts = getattr(opponent, "boosts", None) or {}
        # Speed and accuracy/evasion are not represented by damage ratios above.
        speed_edge = np.log(
            _boost_multiplier(ally_boosts.get("spe", 0))
            / _boost_multiplier(opponent_boosts.get("spe", 0))
        )
        accuracy_edge = np.log(
            _boost_multiplier(ally_boosts.get("accuracy", 0), accuracy=True)
            / _boost_multiplier(opponent_boosts.get("evasion", 0), accuracy=True)
        )
        return float(ally_offense - opponent_offense + 0.25 * speed_edge + 0.15 * accuracy_edge)

    @staticmethod
    def _status_potential(battle: AbstractBattle | None) -> float:
        """Reward useful opponent status conditions once, while they persist."""
        if battle is None:
            return 0.0
        weights = {
            "brn": 0.6, "burn": 0.6,
            "par": 0.7, "paralysis": 0.7,
            "psn": 0.35, "poison": 0.35,
            "tox": 0.65, "toxic": 0.65,
            "slp": 0.9, "sleep": 0.9,
            "frz": 0.9, "freeze": 0.9,
        }
        total = 0.0
        for pokemon in (getattr(battle, "opponent_team", {}) or {}).values():
            status = getattr(pokemon, "status", None)
            if status is not None:
                total += weights.get(_condition_name(status), 0.0)
        return total

    @staticmethod
    def _condition_value(conditions, name: str, max_layers: int = 1) -> float:
        for condition, value in (conditions or {}).items():
            if _condition_name(condition) == name:
                if value is False:
                    return 0.0
                try:
                    count = float(value)
                except (TypeError, ValueError):
                    count = 1.0
                if count <= 0:
                    return 0.0
                return min(float(max_layers), max(1.0, count))
        return 0.0

    @staticmethod
    def _grounded_hazard_coverage(battle: AbstractBattle, *, toxic: bool = False) -> float:
        known_opponents = list((getattr(battle, "opponent_team", {}) or {}).values())
        if not known_opponents:
            return 0.5  # Team is mostly hidden at the start of random battles.

        affected = 0
        for pokemon in known_opponents:
            grounded = getattr(pokemon, "is_grounded", None)
            if grounded is None:
                types = {
                    _condition_name(getattr(pokemon, type_attr, ""))
                    for type_attr in ("type_1", "type_2")
                }
                ability = _condition_name(getattr(pokemon, "ability", ""))
                grounded = "flying" not in types and ability != "levitate"
            if toxic and ("steel" in {
                _condition_name(getattr(pokemon, attr, "")) for attr in ("type_1", "type_2")
            } or "poison" in {
                _condition_name(getattr(pokemon, attr, "")) for attr in ("type_1", "type_2")
            }):
                continue
            affected += bool(grounded)
        unknown_slots = max(0, 6 - len(known_opponents))
        # Estimate unknown team members at 50% susceptibility, then use exact
        # susceptibility for the opponents already revealed.
        return (affected + 0.5 * unknown_slots) / 6

    @classmethod
    def _field_effect_potential(cls, battle: AbstractBattle | None) -> float:
        """Value useful hazards, screens, and speed control currently in play."""
        if battle is None:
            return 0.0
        own = getattr(battle, "side_conditions", {}) or {}
        opponent = getattr(battle, "opponent_side_conditions", {}) or {}

        potential = 0.0
        # Hazards are valued by their layers and estimated coverage of the
        # opponent's known team. Stealth Rock can affect any known team member.
        known_opponents = len(getattr(battle, "opponent_team", {}) or {})
        rock_coverage = (known_opponents + 0.75 * max(0, 6 - known_opponents)) / 6
        potential += 0.45 * cls._condition_value(opponent, "stealthrock") * rock_coverage
        potential += 0.30 * cls._condition_value(opponent, "spikes", 3) * cls._grounded_hazard_coverage(battle)
        potential += 0.30 * cls._condition_value(opponent, "toxicspikes", 2) * cls._grounded_hazard_coverage(battle, toxic=True)
        potential += 0.40 * cls._condition_value(opponent, "stickyweb") * cls._grounded_hazard_coverage(battle)

        # These are our side conditions: defenses and speed control for our team.
        potential += 0.35 * cls._condition_value(own, "reflect")
        potential += 0.35 * cls._condition_value(own, "lightscreen")
        potential += 0.50 * cls._condition_value(own, "auroraveil")
        potential += 0.45 * cls._condition_value(own, "tailwind")
        return potential

    @staticmethod
    def _hazard_entry_bonus(prior_battle, battle, opponent_damage: float) -> float:
        """Add a small bonus when hazards coincide with a damaging switch-in."""
        if prior_battle is None or opponent_damage <= 0:
            return 0.0
        previous_active = getattr(prior_battle, "opponent_active_pokemon", None)
        current_active = getattr(battle, "opponent_active_pokemon", None)
        if previous_active is None or current_active is None:
            return 0.0
        previous_species = _condition_name(getattr(previous_active, "species", ""))
        current_species = _condition_name(getattr(current_active, "species", ""))
        if not previous_species or previous_species == current_species:
            return 0.0

        hazards = getattr(battle, "opponent_side_conditions", {}) or {}
        has_hazards = any(
            ShowdownEnvironment._condition_value(hazards, name, layers) > 0
            for name, layers in (("stealthrock", 1), ("spikes", 3), ("toxicspikes", 2), ("stickyweb", 1))
        )
        return 0.5 * opponent_damage if has_hazards else 0.0

    def _opponent_type_multiplier(self, opponent, target) -> float:
        """Return the best type matchup the opponent can have against our active Pokémon."""
        opponent_types = [
            pokemon_type
            for pokemon_type in (
                getattr(opponent, "type_1", None),
                getattr(opponent, "type_2", None),
            )
            if pokemon_type is not None
        ]
        if not opponent_types or target.type_1 is None:
            return 1.0

        physical_boost = self._stat_stage_damage_multiplier("PHYSICAL", opponent, target)
        special_boost = self._stat_stage_damage_multiplier("SPECIAL", opponent, target)
        strongest_stat_boost = max(physical_boost, special_boost)

        multipliers = []
        for attacking_type in opponent_types:
            try:
                type_multiplier = float(
                        attacking_type.damage_multiplier(
                            target.type_1,
                            getattr(target, "type_2", None),
                            type_chart=GEN_DATA.type_chart,
                        )
                    )
                multipliers.append(type_multiplier * strongest_stat_boost)
            except Exception:
                continue

        # Type effectiveness and the stronger physical/special stat matchup are
        # combined; the opponent's two types do not stack because one move is used.
        return max(multipliers, default=1.0)

    # =========================================================
    # Reward Function
    # =========================================================
    #  reward for the current action
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Reward based on HP, fainted Pokémon, and victory outcomes.
        Inspired by SimpleRLPlayer reward_computing_helper.
        """
        prior_battle = self._get_prior_battle(battle)

        if battle is None:
            return 0.0
        # active_matchup_multiplier = 1.0
            
        # if battle.active_pokemon and battle.opponent_active_pokemon:
        #     active_matchup_multiplier = self._opponent_type_multiplier(
        #         battle.active_pokemon,
        #                         battle.opponent_active_pokemon,
        #                     )
        active_damage_multiplier = 1.0
        opponent_damage_multiplier = 1.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            active_damage_multiplier = self._opponent_type_multiplier(
                battle.active_pokemon,
                battle.opponent_active_pokemon,
            )
            opponent_damage_multiplier = self._opponent_type_multiplier(
                battle.opponent_active_pokemon,
                battle.active_pokemon,
            )

        multDiff = (active_damage_multiplier - opponent_damage_multiplier)

        #Current total ally team HP
        ally_hp = np.sum([m.current_hp_fraction for m in battle.team.values()])
        #Current total of current known opponent team HP
        opp_hp = np.sum([m.current_hp_fraction for m in battle.opponent_team.values()])
        #Weighted difference of above totals
        hp_diff = ally_hp - opp_hp

        #Current number of faints on allied team
        ally_fainted = sum(m.fainted for m in battle.team.values())
        #Current number of faints on opponent team
        opp_fainted = sum(m.fainted for m in battle.opponent_team.values())
        #Weighted difference of above totals
        faint_diff = (opp_fainted - ally_fainted)

        if prior_battle:
            #Last turn total ally team HP
            prev_ally_hp = np.sum([m.current_hp_fraction for m in prior_battle.team.values()])
            #Last turn total of current known Opponent team HP 
            prev_opp_hp = np.sum([m.current_hp_fraction for m in prior_battle.opponent_team.values()])
            opponent_damage = max(0.0, prev_opp_hp - opp_hp)
            ally_damage = max(0.0, prev_ally_hp - ally_hp)
            ally_healing = max(0.0, ally_hp - prev_ally_hp)
            # Reward each recovered HP-fraction point at 1.25x the value of
            # dealing one point of damage to the opponent.
            hp_delta = opponent_damage - ally_damage + 1.25 * ally_healing

            # Potential differences reward useful changes once, rather than
            # paying repeatedly for effects that persist across turns.
            stat_bonus = 0.25 * (
                self._stat_effect_potential(battle)
                - self._stat_effect_potential(prior_battle)
            )
            status_bonus = (
                self._status_potential(battle)
                - self._status_potential(prior_battle)
            )
            field_bonus = (
                self._field_effect_potential(battle)
                - self._field_effect_potential(prior_battle)
            )
            hazard_bonus = self._hazard_entry_bonus(
                prior_battle, battle, opponent_damage
            )
        else:
            hp_delta = 0.0
            stat_bonus = 0.0
            status_bonus = 0.0
            field_bonus = 0.0
            hazard_bonus = 0.0

        # reward for win / loss
        victory_bonus = 0.0
        if battle.finished:
            if battle.won:
                victory_bonus += 1000.0
            elif battle.lost:
                victory_bonus -= 50.0

        # 
        reward = (
            multDiff
            + hp_delta
            + (0.5 * hp_diff)
            + (6.0 * faint_diff)
            + stat_bonus
            + status_bonus
            + field_bonus
            + hazard_bonus
        )
        return float(np.clip(reward, -40.0, 40.0) + victory_bonus)

    # =========================================================
    # Observation space
    # =========================================================
    def _observation_size(self) -> int:
        """
        Embedding structure:
            4x base powers
            4x type multipliers
            7x active ally stat multipliers
            7x active opponent stat multipliers
            6x active ally status flags
            6x active opponent status flags
            2x (ally_fainted/6, opp_fainted/6)
            2x (ally_total_hp, opp_total_hp)
            1x best opponent type/stat multiplier vs. active ally
            5x opponent type/stat multipliers vs. available switches
            5x best known move multipliers for available switches
        = 49 features
        """
        return 49

        # Current State Value
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        SB3-style compact embedding of the battle state.
        Combines per-move offensive info with overall HP context.
        """
        # Update _observation_size() based on what you want to use
        obs = np.zeros(self._observation_size(), dtype=np.float32)

        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            return obs

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        moves_base_power = np.zeros(4, dtype=np.float32)
        moves_dmg_multiplier = np.ones(4, dtype=np.float32)

        for i, move in enumerate(battle.available_moves[:4]):
            moves_base_power[i] = float((move.base_power or 0) / 100.0)
            moves_dmg_multiplier[i] = self._move_damage_multiplier(battle, move, opp)

        def stat_multipliers(pokemon) -> np.ndarray:
            boosts = getattr(pokemon, "boosts", None) or {}
            return np.asarray(
                [
                    _boost_multiplier(
                        boosts.get(stat, 0), accuracy=stat in ("accuracy", "evasion")
                    )
                    for stat in _BATTLE_STATS
                ],
                dtype=np.float32,
            )

        ally_stat_multipliers = stat_multipliers(active)
        opp_stat_multipliers = stat_multipliers(opp)
        ally_status = _status_embedding(active)
        opp_status = _status_embedding(opp)

        opponent_dmg_multiplier = self._opponent_type_multiplier(opp, active)

        # Keep five fixed switch slots so the observation shape is stable.
        # Unavailable slots use neutral multipliers.
        switch_threat_multipliers = np.ones(5, dtype=np.float32)
        switch_best_move_multipliers = np.ones(5, dtype=np.float32)
        for i, switch in enumerate(battle.available_switches[:5]):
            switch_threat_multipliers[i] = self._opponent_type_multiplier(opp, switch)
            switch_moves = getattr(switch, "moves", {}).values()
            switch_best_move_multipliers[i] = max(
                (
                    self._calculate_damage_multiplier(move, switch, opp)
                    for move in switch_moves
                    if getattr(move, "base_power", 0) > 0
                ),
                default=1.0,
            )

        ally_hp_total = np.sum([m.current_hp_fraction for m in battle.team.values()]) / 6.0
        opp_hp_total = np.sum([m.current_hp_fraction for m in battle.opponent_team.values()]) / 6.0
        ally_fainted = len([m for m in battle.team.values() if m.fainted]) / 6.0
        opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted]) / 6.0

        obs = np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            ally_stat_multipliers,
            opp_stat_multipliers,
            ally_status,
            opp_status,
            np.array([ally_fainted, opp_fainted, ally_hp_total, opp_hp_total], dtype=np.float32),
            np.array([opponent_dmg_multiplier], dtype=np.float32),
            switch_threat_multipliers,
            switch_best_move_multipliers,
        ])

        return obs.astype(np.float32)

    # =========================================================
    # Additional info (logging)
    # =========================================================
    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["turns"] = self.battle1.turn
            info[agent]["active_damage_multiplier"] = self._active_damage_multiplier
            info[agent]["opponent_damage_multiplier"] = self._opponent_damage_multiplier
        return info



########################################
# DO NOT EDIT BELOW THIS LINE
########################################

class SingleShowdownWrapper(SingleAgentWrapper):
    """
    Wrapper for single-agent training against specified opponents.
    """

    def __init__(self, team_type: str = "random", opponent_type: str = "random", evaluation: bool = False):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"
        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)
        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")
        bot_teams = {}
        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8") as file:
                    bot_teams[team_file[:-4]] = file.read()
        return bot_teams.get(team_type, None)
