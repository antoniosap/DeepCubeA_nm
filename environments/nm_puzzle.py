from typing import List, Tuple, Union
import numpy as np
import ast
import torch.nn as nn

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State
from random import randrange
from .visualizer import render
import logging


class NMPuzzleState(State):
    __slots__ = ['tiles', 'hash']

    def __init__(self, tiles: np.ndarray):
        self.tiles: np.ndarray = tiles
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.tiles.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)


class NMPuzzle(Environment):
    moves: List[str] = ['U', 'D', 'L', 'R']
    moves_rev: List[str] = ['D', 'U', 'R', 'L']

    def __init__(self, dim_r: int, dim_c: int = None):
        super().__init__()

        self.dim_r: int = dim_r
        if self.dim_r <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int

        self.dim_c: int = dim_c

        # Solved state
        if dim_c is None:
            self.goal_tiles: np.ndarray = np.concatenate((np.arange(1, self.dim_r * self.dim_r), [0])).astype(self.dtype)
            # Next state ops
            self.swap_zero_idxs: np.ndarray = self._get_swap_zero_idxs_n(self.dim_r)
        else:
            self.goal_tiles: np.ndarray = np.concatenate((np.arange(1, self.dim_r * self.dim_c), [0])).astype(self.dtype)
            # Next state ops
            self.swap_zero_idxs: np.ndarray = self._get_swap_zero_idxs_nm(self.dim_r, self.dim_c)

    def next_state(self, states: List[NMPuzzleState], action: int) -> Tuple[List[NMPuzzleState], List[float]]:
        # initialize

        states_np = np.stack([x.tiles for x in states], axis=0)
        states_next_np: np.ndarray = states_np.copy()

        # get zero indicies
        z_idxs: np.ndarray
        _, z_idxs = np.where(states_next_np == 0)

        # get next state
        states_next_np, _, transition_costs = self._move_np(states_np, z_idxs, action)

        # make states
        states_next: List[NMPuzzleState] = [NMPuzzleState(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[NMPuzzleState], action: int) -> List[NMPuzzleState]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[NMPuzzleState], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_tiles.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[NMPuzzleState] = [NMPuzzleState(self.goal_tiles.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[NMPuzzleState]) -> np.ndarray:
        states_np = np.stack([state.tiles for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_tiles, 0))
        solved = np.all(is_equal, axis=1)

        return np.all(is_equal, axis=1)

    def state_to_nnet_input(self, states: List[NMPuzzleState]) -> List[np.ndarray]:
        states_np = np.stack([x.tiles for x in states], axis=0)

        representation = [states_np.astype(self.dtype)]

        return representation

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        state_dim: int = self.dim_r * self.dim_c
        nnet = ResnetModel(state_dim, self.dim_r * self.dim_c, 5000, 1000, 4, 1, True)

        return nnet

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[NMPuzzleState],
                                                                                          List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Get z_idxs
        z_idxs: np.ndarray
        _, z_idxs = np.where(states_np == 0)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        while np.max(num_back_moves < scramble_nums):
            idxs: np.ndarray = np.where((num_back_moves < scramble_nums))[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], z_idxs[idxs], _ = self._move_np(states_np[idxs], z_idxs[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1

        states: List[NMPuzzleState] = [NMPuzzleState(x) for x in list(states_np)]

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.tiles for state in states])

        # Get z_idxs
        z_idxs: np.ndarray
        _, z_idxs = np.where(states_np == 0)

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, _, tc_move = self._move_np(states_np, z_idxs, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(NMPuzzleState(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _get_swap_zero_idxs_n(self, n: int) -> np.ndarray:
        swap_zero_idxs: np.ndarray = np.zeros((n ** 2, len(NMPuzzle.moves)), dtype=self.dtype)
        for moveIdx, move in enumerate(NMPuzzle.moves):
            for i in range(n):
                for j in range(n):
                    z_idx = np.ravel_multi_index((i, j), (n, n))

                    state = np.ones((n, n), dtype=np.int)
                    state[i, j] = 0

                    is_eligible: bool = False
                    if move == 'U':
                        is_eligible = i < (n - 1)
                    elif move == 'D':
                        is_eligible = i > 0
                    elif move == 'L':
                        is_eligible = j < (n - 1)
                    elif move == 'R':
                        is_eligible = j > 0

                    if is_eligible:
                        swap_i: int = -1
                        swap_j: int = -1
                        if move == 'U':
                            swap_i = i + 1
                            swap_j = j
                        elif move == 'D':
                            swap_i = i - 1
                            swap_j = j
                        elif move == 'L':
                            swap_i = i
                            swap_j = j + 1
                        elif move == 'R':
                            swap_i = i
                            swap_j = j - 1

                        swap_zero_idxs[z_idx, moveIdx] = np.ravel_multi_index((swap_i, swap_j), (n, n))
                    else:
                        swap_zero_idxs[z_idx, moveIdx] = z_idx

        return swap_zero_idxs

    def _get_swap_zero_idxs_nm(self, n: int, m: int) -> np.ndarray:
        swap_zero_idxs: np.ndarray = np.zeros((n * m, len(NMPuzzle.moves)), dtype=self.dtype)
        for moveIdx, move in enumerate(NMPuzzle.moves):
            for i in range(n):
                for j in range(m):
                    z_idx = np.ravel_multi_index((i, j), (n, m))

                    state = np.ones((n, m), dtype=np.int)
                    state[i, j] = 0

                    is_eligible: bool = False
                    if move == 'U':
                        is_eligible = i < (n - 1)
                    elif move == 'D':
                        is_eligible = i > 0
                    elif move == 'L':
                        is_eligible = j < (m - 1)
                    elif move == 'R':
                        is_eligible = j > 0

                    if is_eligible:
                        swap_i: int = -1
                        swap_j: int = -1
                        if move == 'U':
                            swap_i = i + 1
                            swap_j = j
                        elif move == 'D':
                            swap_i = i - 1
                            swap_j = j
                        elif move == 'L':
                            swap_i = i
                            swap_j = j + 1
                        elif move == 'R':
                            swap_i = i
                            swap_j = j - 1

                        swap_zero_idxs[z_idx, moveIdx] = np.ravel_multi_index((swap_i, swap_j), (n, m))
                    else:
                        swap_zero_idxs[z_idx, moveIdx] = z_idx

        return swap_zero_idxs

    def _move_np(self, states_np: np.ndarray, z_idxs: np.array,
                 action: int) -> Tuple[np.ndarray, np.array, List[float]]:
        states_next_np: np.ndarray = states_np.copy()

        # get index to swap with zero
        state_idxs: np.ndarray = np.arange(0, states_next_np.shape[0])
        swap_z_idxs: np.ndarray = self.swap_zero_idxs[z_idxs, action]

        # swap zero with adjacent tile
        states_next_np[state_idxs, z_idxs] = states_np[state_idxs, swap_z_idxs]
        states_next_np[state_idxs, swap_z_idxs] = 0

        # transition costs
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, swap_z_idxs, transition_costs

    @staticmethod
    def load_state(state_2d):
        return np.array(state_2d, dtype=np.int).flatten()

    @staticmethod
    def entry_user_state() -> List[State]:
        # example
        # EASY [[ 1, 2, 3, 4, 5, 6, 7, 8],[9, 10, 11, 12, 13, 14, 15, 16],[ 17, 18, 19, 20, 21, 22, 23, 24],[25, 26, 27, 0, 28, 29, 30, 31]]
        # OK DIFFICULT [[ 1,  2, 11,  3,  4,  6, 16,  7],[10, 25, 13, 12,  5,  0, 14,  8],[ 9, 20, 18, 27, 22, 23, 15, 24],[17, 26, 19, 28, 21, 29, 30, 31]]
        # NO [[ 10, 11, 3,  4,  12, 6, 16, 7 ],[9, 25, 13, 18, 22,  5, 14,  8],[ 17, 20, 21, 28, 27, 15, 24, 31],[26, 19, 2, 1, 29, 23, 30, 0]]
        # NO [[ 1, 17, 2, 18, 3, 19, 4, 20],[21, 5, 22, 6, 23, 7, 24, 8],[ 9, 25, 10, 26, 11, 27, 12, 28],[29, 13, 30, 14, 31, 15, 0, 16]]
        # NO [[ 1, 2, 3, 4, 5, 6, 7, 8],[9, 10, 11, 12, 13, 14, 15, 16],[ 17, 18, 19, 20, 21, 22, 23, 24],[28, 26, 25, 29, 30, 27, 31, 0]]
        #
        # entry = input('entry board as [[...],[...],...] 4 rows, 8 colums: ')
        # entry = '[[ 1,  2, 11,  3,  4,  6, 16,  7],[10, 25, 13, 12,  5,  0, 14,  8],[ 9, 20, 18, 27, 22, 23, 15, 24],[17, 26, 19, 28, 21, 29, 30, 31]]'
        # entry = '[[ 10, 11, 3,  4,  12, 6, 16, 7 ],[9, 25, 13, 18, 22,  5, 14,  8],[ 17, 20, 21, 28, 27, 15, 24, 31],[26, 19, 2, 1, 29, 23, 30, 0]]'
        entry = '[[10, 11, 3, 4, 12, 6, 16, 7],[ 9, 25, 13, 18, 22, 5, 14, 8],[17, 20, 21, 28, 27, 15, 24, 31],[26, 19, 2, 1, 29, 23, 30, 0]]'
        init_state = ast.literal_eval(entry)
        return [NMPuzzleState(NMPuzzle.load_state(init_state))]

# ecco la soluzione:
# sequenza critica: distanza 24 esito: non risolta in 48h da IDA*
#         init_state = (
#             (1, 2, 11, 3, 4, 6, 16, 7),
#             (10, 25, 13, 12, 5, 0, 14, 8),
#             (9, 20, 18, 27, 22, 23, 15, 24),
#             (17, 26, 19, 28, 21, 29, 30, 31),
#         )
# 2021-01-16 17:57:02,999 INFO ========================================
# 2021-01-16 18:49:57,229 INFO Times - pop: 0.01, expand: 0.12, check: 0.08, heur: 0.68, add: 0.00, itr: 0.89, num_itrs: 185
# 2021-01-16 18:49:57,230 INFO State: 0, SolnCost: 38.00, # Moves: 38, # Nodes Gen: 7,340, Time: 1.11
# 2021-01-16 18:49:57,230 INFO ---------------------------------------- BATCH NR: 1
# 2021-01-16 18:49:57,230 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', '16', ' 7']
# 2021-01-16 18:49:57,230 INFO ['10', '25', '13', '12', ' 5', '  ', '14', ' 8']
# 2021-01-16 18:49:57,230 INFO [' 9', '20', '18', '27', '22', '23', '15', '24']
# 2021-01-16 18:49:57,230 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,230 INFO ---------------------------------------- 1 (<-- LEFT)
# 2021-01-16 18:49:57,230 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', '16', ' 7']
# 2021-01-16 18:49:57,230 INFO ['10', '25', '13', '12', ' 5', '14', '  ', ' 8']
# 2021-01-16 18:49:57,231 INFO [' 9', '20', '18', '27', '22', '23', '15', '24']
# 2021-01-16 18:49:57,231 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,231 INFO ---------------------------------------- 2 (DOWN vvv)
# 2021-01-16 18:49:57,231 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', '  ', ' 7']
# 2021-01-16 18:49:57,231 INFO ['10', '25', '13', '12', ' 5', '14', '16', ' 8']
# 2021-01-16 18:49:57,231 INFO [' 9', '20', '18', '27', '22', '23', '15', '24']
# 2021-01-16 18:49:57,231 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,231 INFO ---------------------------------------- 3 (<-- LEFT)
# 2021-01-16 18:49:57,231 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', '  ']
# 2021-01-16 18:49:57,231 INFO ['10', '25', '13', '12', ' 5', '14', '16', ' 8']
# 2021-01-16 18:49:57,231 INFO [' 9', '20', '18', '27', '22', '23', '15', '24']
# 2021-01-16 18:49:57,232 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,232 INFO ---------------------------------------- 4 (UP ^^^)
# 2021-01-16 18:49:57,232 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,232 INFO ['10', '25', '13', '12', ' 5', '14', '16', '  ']
# 2021-01-16 18:49:57,232 INFO [' 9', '20', '18', '27', '22', '23', '15', '24']
# 2021-01-16 18:49:57,232 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,232 INFO ---------------------------------------- 5 (RIGHT -->)
# 2021-01-16 18:49:57,232 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,232 INFO ['10', '25', '13', '12', ' 5', '14', '  ', '16']
# 2021-01-16 18:49:57,232 INFO [' 9', '20', '18', '27', '22', '23', '15', '24']
# 2021-01-16 18:49:57,232 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,232 INFO ---------------------------------------- 6 (UP ^^^)
# 2021-01-16 18:49:57,232 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,233 INFO ['10', '25', '13', '12', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,233 INFO [' 9', '20', '18', '27', '22', '23', '  ', '24']
# 2021-01-16 18:49:57,233 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,233 INFO ---------------------------------------- 7 (RIGHT -->)
# 2021-01-16 18:49:57,233 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,233 INFO ['10', '25', '13', '12', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,233 INFO [' 9', '20', '18', '27', '22', '  ', '23', '24']
# 2021-01-16 18:49:57,233 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,233 INFO ---------------------------------------- 8 (RIGHT -->)
# 2021-01-16 18:49:57,233 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,233 INFO ['10', '25', '13', '12', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,233 INFO [' 9', '20', '18', '27', '  ', '22', '23', '24']
# 2021-01-16 18:49:57,234 INFO ['17', '26', '19', '28', '21', '29', '30', '31']
# 2021-01-16 18:49:57,234 INFO ---------------------------------------- 9 (UP ^^^)
# 2021-01-16 18:49:57,234 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,234 INFO ['10', '25', '13', '12', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,234 INFO [' 9', '20', '18', '27', '21', '22', '23', '24']
# 2021-01-16 18:49:57,234 INFO ['17', '26', '19', '28', '  ', '29', '30', '31']
# 2021-01-16 18:49:57,234 INFO ---------------------------------------- 10 (RIGHT -->)
# 2021-01-16 18:49:57,234 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,234 INFO ['10', '25', '13', '12', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,234 INFO [' 9', '20', '18', '27', '21', '22', '23', '24']
# 2021-01-16 18:49:57,234 INFO ['17', '26', '19', '  ', '28', '29', '30', '31']
# 2021-01-16 18:49:57,234 INFO ---------------------------------------- 11 (DOWN vvv)
# 2021-01-16 18:49:57,235 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,235 INFO ['10', '25', '13', '12', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,235 INFO [' 9', '20', '18', '  ', '21', '22', '23', '24']
# 2021-01-16 18:49:57,235 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,235 INFO ---------------------------------------- 12 (DOWN vvv)
# 2021-01-16 18:49:57,235 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,235 INFO ['10', '25', '13', '  ', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,235 INFO [' 9', '20', '18', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,235 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,235 INFO ---------------------------------------- 13 (RIGHT -->)
# 2021-01-16 18:49:57,235 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,235 INFO ['10', '25', '  ', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,235 INFO [' 9', '20', '18', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,235 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,236 INFO ---------------------------------------- 14 (UP ^^^)
# 2021-01-16 18:49:57,236 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,236 INFO ['10', '25', '18', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,236 INFO [' 9', '20', '  ', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,236 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,236 INFO ---------------------------------------- 15 (RIGHT -->)
# 2021-01-16 18:49:57,236 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,236 INFO ['10', '25', '18', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,236 INFO [' 9', '  ', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,236 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,236 INFO ---------------------------------------- 16 (DOWN vvv)
# 2021-01-16 18:49:57,236 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,237 INFO ['10', '  ', '18', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,237 INFO [' 9', '25', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,237 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,237 INFO ---------------------------------------- 17 (<-- LEFT)
# 2021-01-16 18:49:57,237 INFO [' 1', ' 2', '11', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,237 INFO ['10', '18', '  ', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,237 INFO [' 9', '25', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,237 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,237 INFO ---------------------------------------- 18 (DOWN vvv)
# 2021-01-16 18:49:57,237 INFO [' 1', ' 2', '  ', ' 3', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,237 INFO ['10', '18', '11', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,237 INFO [' 9', '25', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,238 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,238 INFO ---------------------------------------- 19 (<-- LEFT)
# 2021-01-16 18:49:57,238 INFO [' 1', ' 2', ' 3', '  ', ' 4', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,238 INFO ['10', '18', '11', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,238 INFO [' 9', '25', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,238 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,238 INFO ---------------------------------------- 20 (<-- LEFT)
# 2021-01-16 18:49:57,238 INFO [' 1', ' 2', ' 3', ' 4', '  ', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,238 INFO ['10', '18', '11', '13', ' 5', '14', '15', '16']
# 2021-01-16 18:49:57,238 INFO [' 9', '25', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,238 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,238 INFO ---------------------------------------- 21 (UP ^^^)
# 2021-01-16 18:49:57,238 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,239 INFO ['10', '18', '11', '13', '  ', '14', '15', '16']
# 2021-01-16 18:49:57,239 INFO [' 9', '25', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,239 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,239 INFO ---------------------------------------- 22 (RIGHT -->)
# 2021-01-16 18:49:57,239 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,239 INFO ['10', '18', '11', '  ', '13', '14', '15', '16']
# 2021-01-16 18:49:57,239 INFO [' 9', '25', '20', '12', '21', '22', '23', '24']
# 2021-01-16 18:49:57,239 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,239 INFO ---------------------------------------- 23 (UP ^^^)
# 2021-01-16 18:49:57,239 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,239 INFO ['10', '18', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,239 INFO [' 9', '25', '20', '  ', '21', '22', '23', '24']
# 2021-01-16 18:49:57,240 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,240 INFO ---------------------------------------- 24 (RIGHT -->)
# 2021-01-16 18:49:57,240 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,240 INFO ['10', '18', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,240 INFO [' 9', '25', '  ', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,240 INFO ['17', '26', '19', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,240 INFO ---------------------------------------- 25 (UP ^^^)
# 2021-01-16 18:49:57,240 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,240 INFO ['10', '18', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,240 INFO [' 9', '25', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,240 INFO ['17', '26', '  ', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,240 INFO ---------------------------------------- 26 (RIGHT -->)
# 2021-01-16 18:49:57,241 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,241 INFO ['10', '18', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,241 INFO [' 9', '25', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,241 INFO ['17', '  ', '26', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,241 INFO ---------------------------------------- 27 (DOWN vvv)
# 2021-01-16 18:49:57,241 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,241 INFO ['10', '18', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,241 INFO [' 9', '  ', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,241 INFO ['17', '25', '26', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,241 INFO ---------------------------------------- 28 (DOWN vvv)
# 2021-01-16 18:49:57,241 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,241 INFO ['10', '  ', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,242 INFO [' 9', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,242 INFO ['17', '25', '26', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,242 INFO ---------------------------------------- 29 (RIGHT -->)
# 2021-01-16 18:49:57,242 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,242 INFO ['  ', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,242 INFO [' 9', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,242 INFO ['17', '25', '26', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,242 INFO ---------------------------------------- 30 (UP ^^^)
# 2021-01-16 18:49:57,242 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,242 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,242 INFO ['  ', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,242 INFO ['17', '25', '26', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,243 INFO ---------------------------------------- 31 (UP ^^^)
# 2021-01-16 18:49:57,243 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,243 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,243 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,243 INFO ['  ', '25', '26', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,243 INFO ---------------------------------------- 32 (<-- LEFT)
# 2021-01-16 18:49:57,243 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,243 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,243 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,243 INFO ['25', '  ', '26', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,243 INFO ---------------------------------------- 33 (<-- LEFT)
# 2021-01-16 18:49:57,243 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,243 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,244 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,244 INFO ['25', '26', '  ', '27', '28', '29', '30', '31']
# 2021-01-16 18:49:57,244 INFO ---------------------------------------- 34 (<-- LEFT)
# 2021-01-16 18:49:57,244 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,244 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,244 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,244 INFO ['25', '26', '27', '  ', '28', '29', '30', '31']
# 2021-01-16 18:49:57,244 INFO ---------------------------------------- 35 (<-- LEFT)
# 2021-01-16 18:49:57,244 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,244 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,244 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,244 INFO ['25', '26', '27', '28', '  ', '29', '30', '31']
# 2021-01-16 18:49:57,244 INFO ---------------------------------------- 36 (<-- LEFT)
# 2021-01-16 18:49:57,245 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,245 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,245 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,245 INFO ['25', '26', '27', '28', '29', '  ', '30', '31']
# 2021-01-16 18:49:57,245 INFO ---------------------------------------- 37 (<-- LEFT)
# 2021-01-16 18:49:57,245 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,245 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,245 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,245 INFO ['25', '26', '27', '28', '29', '30', '  ', '31']
# 2021-01-16 18:49:57,245 INFO ---------------------------------------- 38 (<-- LEFT)
# 2021-01-16 18:49:57,246 INFO [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
# 2021-01-16 18:49:57,246 INFO [' 9', '10', '11', '12', '13', '14', '15', '16']
# 2021-01-16 18:49:57,246 INFO ['17', '18', '19', '20', '21', '22', '23', '24']
# 2021-01-16 18:49:57,246 INFO ['25', '26', '27', '28', '29', '30', '31', '  ']
# 2021-01-16 18:49:57,246 INFO ========================================
