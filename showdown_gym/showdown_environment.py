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
        # These values are logging metadata read during the first reset, before
        # an active battle has been observed.
        self._active_damage_multiplier = 1.0
        self._opponent_damage_multiplier = 1.0

    # =========================================================
    # Action space
    # =========================================================
    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        # Keep the first ten Showdown action IDs unchanged: six switch slots
        # (0-5) and four move slots (6-9).
        return 10

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        Actions 0-5 select switch slots and actions 6-9 select moves, matching
        the underlying Showdown action IDs. Negative and out-of-range values
        select the default action.

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        action = int(action)
        if action < 0 or action >= 10:
            return np.int64(-2)
        return np.int64(action)

    def get_available_action_mask(self) -> np.ndarray:
        """Report all ten actions as available; this environment applies no mask."""
        return np.ones(10, dtype=np.float32)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def _move_damage_multiplier(self, battle: AbstractBattle, move, opponent) -> float:
        """
        Combine the move's type effectiveness with the relevant offensive and defensive stat boosts.
        """
        return self._calculate_damage_multiplier(
            move, battle.active_pokemon, opponent
        )

    def _calculate_damage_multiplier(self, move, attacker, defender) -> float:
        """Calculate expected damage multiplier, including effective accuracy."""
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

        # Poke-env move accuracy is generally expressed as a percentage. Treat
        # missing/boolean accuracy as always-hit, and account for accuracy and
        # evasion stages when estimating the chance that the move connects.
        accuracy = getattr(move, "accuracy", None)
        if accuracy is None or isinstance(accuracy, (bool, np.bool_)):
            hit_probability = 1.0
        else:
            accuracy = float(accuracy)
            base_hit_probability = accuracy / 100.0 if accuracy > 1.0 else accuracy
            attacker_boosts = getattr(attacker, "boosts", None) or {}
            defender_boosts = getattr(defender, "boosts", None) or {}
            accuracy_multiplier = _boost_multiplier(
                attacker_boosts.get("accuracy", 0), accuracy=True
            )
            evasion_multiplier = _boost_multiplier(
                defender_boosts.get("evasion", 0), accuracy=True
            )
            hit_probability = np.clip(
                base_hit_probability * accuracy_multiplier / evasion_multiplier,
                0.0,
                1.0,
            )
        multiplier *= float(hit_probability)

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
        """Value active stat changes in the current matchup."""
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
        """Value useful opponent status conditions while they persist."""
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

    def _opponent_type_multiplier(self, opponent, target) -> float:
        """Return the opponent's strongest type and stat multiplier against a target."""
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

    def _active_matchup_potential(self, battle: AbstractBattle) -> float:
        """Value a favorable active matchup using approximate expected damage."""
        ally = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        if ally is None or opponent is None:
            return 0.0

        ally_moves = list(getattr(battle, "available_moves", []) or [])
        if not ally_moves:
            ally_moves = list((getattr(ally, "moves", {}) or {}).values())
        opponent_moves = list((getattr(opponent, "moves", {}) or {}).values())

        def best_expected_damage(attacker, defender, moves):
            return max(
                (
                    float(getattr(move, "base_power", 0) or 0)
                    * self._calculate_damage_multiplier(move, attacker, defender)
                    for move in moves
                    if getattr(move, "base_power", 0) > 0
                ),
                default=100.0,
            )

        ally_attack = best_expected_damage(ally, opponent, ally_moves)
        opponent_attack = best_expected_damage(opponent, ally, opponent_moves)
        if not opponent_moves:
            opponent_attack = 100.0 * self._opponent_type_multiplier(opponent, ally)

        matchup_ratio = max(ally_attack, 1e-6) / max(opponent_attack, 1e-6)
        matchup_score = float(np.clip(np.log2(matchup_ratio), -2.0, 2.0))
        health_weight = min(
            float(getattr(ally, "current_hp_fraction", 0.0)),
            float(getattr(opponent, "current_hp_fraction", 0.0)),
        )
        return 2.0 * health_weight * matchup_score

    # =========================================================
    # Reward Function
    # =========================================================
    #  reward for the current action
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Reward based on HP, fainted Pokémon, and victory outcomes.
        Inspired by SimpleRLPlayer reward_computing_helper.
        """
        if battle is None:
            return 0.0
        prior_battle = self._get_prior_battle(battle)
        status_stat_reward = 0.0
        matchup_reward = 0.0
        if not battle.finished:
            status_stat_reward = 0.25 * 0.99 * (
                0.25 * self._stat_effect_potential(battle)
                + 0.75 * self._status_potential(battle)
            )
            matchup_reward = 0.10 * max(
                0.0, self._active_matchup_potential(battle)
            )

        switch_reward = 0.0
        damage_exchange_reward = 0.0
        if prior_battle:
            # Give an extra one-time nudge when an actual switch improves the
            # matchup. Fainted/forced switch-ins are excluded by requiring the
            # outgoing active Pokémon to still be alive.
            previous_active = getattr(prior_battle, "active_pokemon", None)
            current_active = getattr(battle, "active_pokemon", None)
            if (
                previous_active is not None
                and current_active is not None
                and not getattr(previous_active, "fainted", False)
                and _condition_name(getattr(previous_active, "species", ""))
                != _condition_name(getattr(current_active, "species", ""))
            ):
                old_matchup = self._active_matchup_potential(prior_battle)
                new_matchup = self._active_matchup_potential(battle)
                switch_reward = 0.5 * max(0.0, new_matchup - old_matchup)

            previous_active = getattr(prior_battle, "active_pokemon", None)
            current_active = getattr(battle, "active_pokemon", None)
            previous_opponent = getattr(prior_battle, "opponent_active_pokemon", None)
            current_opponent = getattr(battle, "opponent_active_pokemon", None)
            ally_hp_change = (
                float(getattr(current_active, "current_hp_fraction", 0.0))
                - float(getattr(previous_active, "current_hp_fraction", 0.0))
            )
            opponent_hp_change = (
                float(getattr(current_opponent, "current_hp_fraction", 0.0))
                - float(getattr(previous_opponent, "current_hp_fraction", 0.0))
            )
            # Value damage dealt to the opponent 1.5x, while penalizing our
            # own HP loss (including Substitute) at the base rate.
            damage_exchange_reward = float(
                np.clip(
                    2.0 * (ally_hp_change - 1.5 * opponent_hp_change),
                    -4.0,
                    4.0,
                )
            )
        # Charge every action a time cost so ineffective turns and drawn-out
        # battles are more costly; useful turns must earn enough progress to offset it.
        step_cost = -0.3

        # reward for win / loss
        victory_bonus = 0.0
        if battle.finished:
            if battle.won:
                victory_bonus += 100.0
            elif battle.lost:
                victory_bonus -= 100.0

        return float(
            damage_exchange_reward
            + step_cost
            + status_stat_reward
            + matchup_reward
            + (switch_reward if prior_battle else 0.0)
            + victory_bonus
        )

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
            5x opponent type/stat multipliers vs. available switches
            5x best known move multipliers for available switches
            4x relevant side conditions
        = 52 features
        """
        return 52

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

        own_conditions = getattr(battle, "side_conditions", {}) or {}
        field_effects = np.asarray(
            [
                self._condition_value(own_conditions, "reflect"),
                self._condition_value(own_conditions, "lightscreen"),
                self._condition_value(own_conditions, "auroraveil"),
                self._condition_value(own_conditions, "tailwind"),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            ally_stat_multipliers,
            opp_stat_multipliers,
            ally_status,
            opp_status,
            np.array([ally_fainted, opp_fainted, ally_hp_total, opp_hp_total], dtype=np.float32),
            switch_threat_multipliers,
            switch_best_move_multipliers,
            field_effects,
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
