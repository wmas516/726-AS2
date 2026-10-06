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
        multiplier = 1.0
        active = battle.active_pokemon
        opponent_active = battle.opponent_active_pokemon

        def stage_multiplier(stage: int) -> float:
            if stage >= 0:
                return float((2 + stage) / 2)
            return float(2 / (2 - stage))

        try:
            if move.type and opponent.type_1:
                multiplier = float(
                    move.type.damage_multiplier(
                        opponent.type_1,
                        getattr(opponent, "type_2", None),
                        type_chart=GEN_DATA.type_chart,
                    )
                )
        except Exception:
            multiplier = 1.0

        move_category = getattr(move.category, "name", str(move.category)).upper()
        if move_category == "PHYSICAL":
            multiplier *= stage_multiplier(int(active.boosts.get("atk", 0)) if active is not None else 0)
            opponent_defense = stage_multiplier(int(opponent_active.boosts.get("def", 0)) if opponent_active is not None else 0)
            if opponent_defense != 0:
                multiplier /= opponent_defense
        elif move_category == "SPECIAL":
            multiplier *= stage_multiplier(int(active.boosts.get("spa", 0)) if active is not None else 0)
            opponent_sp_defense = stage_multiplier(int(opponent_active.boosts.get("spd", 0)) if opponent_active is not None else 0)
            if opponent_sp_defense != 0:
                multiplier /= opponent_sp_defense

        return float(multiplier)

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
        faint_diff = (opp_fainted - ally_fainted) * 2.0

        if prior_battle:
            #Last turn total ally team HP
            prev_ally_hp = np.sum([m.current_hp_fraction for m in prior_battle.team.values()])
            #Last turn total of current known Opponent team HP 
            prev_opp_hp = np.sum([m.current_hp_fraction for m in prior_battle.opponent_team.values()])
            #Change of hp between turns of opp - ally 
            # - More + more opp hp lost and/or ally hp gained
            # - More - less opp hp lost and/or ally hp gained
            hp_delta = (prev_opp_hp - opp_hp) - (prev_ally_hp - ally_hp)
        else:
            hp_delta = 0.0

        # reward for win / loss
        victory_bonus = 0.0
        if battle.finished:
            if battle.won:
                victory_bonus += 1000.0
            elif battle.lost:
                victory_bonus -= 50.0

        # 
        reward = 1.0 * hp_delta + 0.5 * hp_diff + faint_diff #+ victory_bonus
        return float(np.clip(reward, -20.0, 20.0) + victory_bonus)

    # =========================================================
    # Observation space
    # =========================================================
    def _observation_size(self) -> int:
        """
        Embedding structure:
            4x base powers
            4x type multipliers
            2x (ally_fainted/6, opp_fainted/6)
            2x (ally_total_hp, opp_total_hp)
        = 12 features
        """
        return 12

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

        ally_hp_total = np.sum([m.current_hp_fraction for m in battle.team.values()]) / 6.0
        opp_hp_total = np.sum([m.current_hp_fraction for m in battle.opponent_team.values()]) / 6.0
        ally_fainted = len([m for m in battle.team.values() if m.fainted]) / 6.0
        opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted]) / 6.0

        obs = np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            np.array([ally_fainted, opp_fainted, ally_hp_total, opp_hp_total], dtype=np.float32),
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
