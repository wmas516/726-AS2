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
        self._last_action = None
        self._blocked_moves = {}
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
            self._last_action = None
            return np.int64(-2)
        self._last_action = action
        return np.int64(action)

    def get_available_action_mask(self) -> np.ndarray:
        """Mask a move after it failed, until the battle state changes."""
        mask = np.ones(10, dtype=np.float32)
        battle = self.battle1
        if battle is None or battle.active_pokemon is None:
            return mask

        active_species = _condition_name(battle.active_pokemon.species)
        current_signature = self._battle_effect_signature(battle)
        available_moves = list(getattr(battle, "available_moves", []) or [])
        for slot, move in enumerate(available_moves[:4]):
            key = (active_species, _condition_name(getattr(move, "id", move)))
            blocked_signature = self._blocked_moves.get(key)
            if blocked_signature is None:
                continue
            if blocked_signature == current_signature:
                mask[6 + slot] = 0.0
            else:
                # The relevant state changed, so the previously failed move
                # may now have a useful effect.
                del self._blocked_moves[key]

        # Never mask every move when there is no switch option; keep the agent
        # able to act if the only remaining choice is a previously failed move.
        if available_moves and not np.any(mask[6 : 6 + min(4, len(available_moves))]):
            switches = list(getattr(battle, "available_switches", []) or [])
            if not switches:
                mask[6 : 6 + min(4, len(available_moves))] = 1.0
        return mask

    def reset(self, seed=None, options=None):
        self._last_action = None
        self._blocked_moves.clear()
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

    def _battle_potential(self, battle: AbstractBattle | None) -> float:
        """Estimate battle advantage for small, discounted reward shaping."""
        if battle is None:
            return 0.0
        ally_fainted = sum(m.fainted for m in battle.team.values())
        opp_fainted = sum(m.fainted for m in battle.opponent_team.values())
        faint_advantage = 0.5 * (opp_fainted - ally_fainted)
        tactical_advantage = (
            0.25 * self._stat_effect_potential(battle)
            + 0.75 * self._status_potential(battle)
            + 0.25 * self._field_effect_potential(battle)
        )
        matchup_advantage = self._active_matchup_potential(battle)
        return float(faint_advantage + tactical_advantage + matchup_advantage)

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

    @staticmethod
    def _battle_effect_signature(battle: AbstractBattle):
        """Capture persistent effects that indicate a move changed the battle."""
        def team_effects(team):
            effects = {}
            for key, pokemon in (team or {}).items():
                effects[key] = (
                    _condition_name(getattr(pokemon, "status", None)),
                    tuple(sorted((str(k), int(v)) for k, v in (getattr(pokemon, "boosts", {}) or {}).items())),
                    tuple(sorted(_condition_name(v) for v in (getattr(pokemon, "effects", {}) or {}).keys())),
                )
            return effects

        def conditions(values):
            signature = []
            for key, value in (values or {}).items():
                name = _condition_name(key)
                try:
                    active = float(value) > 0
                except (TypeError, ValueError):
                    active = value is not False
                if not active:
                    continue
                # Layer counts are meaningful for these hazards; duration
                # counters on screens/Tailwind should not look like new effects.
                layers = str(value) if name in ("spikes", "toxicspikes") else "active"
                signature.append((name, layers))
            return tuple(sorted(signature))

        return (
            _condition_name(getattr(getattr(battle, "active_pokemon", None), "species", "")),
            _condition_name(getattr(getattr(battle, "opponent_active_pokemon", None), "species", "")),
            team_effects(getattr(battle, "team", {})),
            team_effects(getattr(battle, "opponent_team", {})),
            conditions(getattr(battle, "side_conditions", {})),
            conditions(getattr(battle, "opponent_side_conditions", {})),
        )

    def _last_move_had_no_effect(self, prior_battle, battle) -> bool:
        """Detect a move with no observable battle effect, regardless of category."""
        action = self._last_action
        if action is None or not 6 <= action <= 9:
            return False
        moves = list(getattr(prior_battle, "available_moves", []) or [])
        move_index = action - 6
        if move_index >= len(moves):
            return False

        # New status, stat stage, volatile, or side-condition effects count for
        # any move category (including secondary effects from attacks).
        if self._battle_effect_signature(prior_battle) != self._battle_effect_signature(battle):
            return False

        # Damage moves are effective when they lower the same active target's
        # HP. Ignore a different opponent active Pokémon to avoid crediting a
        # newly switched-in Pokémon's starting HP as an effect.
        previous_opponent = getattr(prior_battle, "opponent_active_pokemon", None)
        current_opponent = getattr(battle, "opponent_active_pokemon", None)
        if previous_opponent is not None and current_opponent is not None:
            same_target = (
                _condition_name(getattr(previous_opponent, "species", ""))
                == _condition_name(getattr(current_opponent, "species", ""))
            )
            if same_target and (
                current_opponent.current_hp_fraction
                < previous_opponent.current_hp_fraction
            ):
                return False

        # Healing moves count as effective when the active Pokémon recovers HP.
        previous_active = getattr(prior_battle, "active_pokemon", None)
        current_active = getattr(battle, "active_pokemon", None)
        if previous_active is not None and current_active is not None:
            same_active = (
                _condition_name(getattr(previous_active, "species", ""))
                == _condition_name(getattr(current_active, "species", ""))
            )
            if same_active and (
                current_active.current_hp_fraction
                > previous_active.current_hp_fraction
            ):
                return False

        return True

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
        if prior_battle:
            # Keep gamma aligned with the DoubleDQN configuration (0.99).
            # Scaling the potential keeps shaping small relative to the outcome.
            next_potential = 0.0 if battle.finished else self._battle_potential(battle)
            battle_progress = 0.25 * (
                0.99 * next_potential - self._battle_potential(prior_battle)
            )

            # Pay a small continuing reward for maintaining a favorable active
            # matchup. The potential difference above alone only rewards entry
            # into that position, not staying in it.
            matchup_hold_reward = 0.10 * max(
                0.0, self._active_matchup_potential(battle)
            )

            # Give an extra one-time nudge when an actual switch improves the
            # matchup. Fainted/forced switch-ins are excluded by requiring the
            # outgoing active Pokémon to still be alive.
            previous_active = getattr(prior_battle, "active_pokemon", None)
            current_active = getattr(battle, "active_pokemon", None)
            switch_reward = 0.0
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

            def hp_change(previous_team, current_team, *, known_members_only=False):
                previous = previous_team or {}
                current = current_team or {}
                keys = current.keys() & previous.keys() if known_members_only else current.keys()
                return sum(
                    current[key].current_hp_fraction
                    - previous[key].current_hp_fraction
                    for key in keys
                )

            ally_hp_change = hp_change(prior_battle.team, battle.team)
            # Opponent teams are only revealed gradually; compare shared members
            # so a newly revealed full-health Pokémon is not mistaken for healing.
            opponent_hp_change = hp_change(
                prior_battle.opponent_team,
                battle.opponent_team,
                known_members_only=True,
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
        else:
            battle_progress = 0.0
            damage_exchange_reward = 0.0

        # Charge every action a time cost so ineffective turns and drawn-out
        # battles are more costly; useful turns must earn enough progress to offset it.
        step_cost = -0.3

        no_effect_move_penalty = 0.0
        controlled_battle = (
            str(getattr(battle, "player_username", "")).lower()
            == str(self.rl_agent).lower()
        )
        if (
            prior_battle
            and controlled_battle
            and self._last_move_had_no_effect(prior_battle, battle)
        ):
            # A move with no observable effect should cost more than the
            # ordinary turn cost, regardless of its category.
            no_effect_move_penalty = -1.0
            action = self._last_action
            moves = list(getattr(prior_battle, "available_moves", []) or [])
            if action is not None and 6 <= action <= 9 and action - 6 < len(moves):
                move = moves[action - 6]
                active = getattr(prior_battle, "active_pokemon", None)
                if active is not None:
                    key = (
                        _condition_name(getattr(active, "species", "")),
                        _condition_name(getattr(move, "id", move)),
                    )
                    self._blocked_moves[key] = self._battle_effect_signature(battle)

        # reward for win / loss
        victory_bonus = 0.0
        if battle.finished:
            if battle.won:
                victory_bonus += 100.0
            elif battle.lost:
                victory_bonus -= 100.0

        return float(
            np.clip(battle_progress, -10.0, 10.0)
            + damage_exchange_reward
            + step_cost
            + (matchup_hold_reward if prior_battle else 0.0)
            + (switch_reward if prior_battle else 0.0)
            + no_effect_move_penalty
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
            1x best opponent type/stat multiplier vs. active ally
            5x opponent type/stat multipliers vs. available switches
            5x best known move multipliers for available switches
            8x relevant side conditions and hazards
        = 57 features
        """
        return 57

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

        own_conditions = getattr(battle, "side_conditions", {}) or {}
        opponent_conditions = getattr(battle, "opponent_side_conditions", {}) or {}
        field_effects = np.asarray(
            [
                self._condition_value(own_conditions, "reflect"),
                self._condition_value(own_conditions, "lightscreen"),
                self._condition_value(own_conditions, "auroraveil"),
                self._condition_value(own_conditions, "tailwind"),
                self._condition_value(opponent_conditions, "stealthrock"),
                self._condition_value(opponent_conditions, "spikes", 3) / 3.0,
                self._condition_value(opponent_conditions, "toxicspikes", 2) / 2.0,
                self._condition_value(opponent_conditions, "stickyweb"),
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
            np.array([opponent_dmg_multiplier], dtype=np.float32),
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
