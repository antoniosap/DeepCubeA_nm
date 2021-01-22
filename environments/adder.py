#
# 16.1.2021
#
# https://theorycircuit.com/full-adder-circuit-diagram/
# https://en.wikipedia.org/wiki/Logic_gate
# https://github.com/scale-lab/BLASYS
# https://arxiv.org/pdf/1902.00478.pdf
#
# sintetizzare una rete combinatoria per un addizionatore N cifre binarie
#
#
from typing import List, Tuple, Union
import numpy as np
import ast
import torch.nn as nn

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State
from random import randrange
from .visualizer import render
import logging


# full adder schematic:
#                      block 1                                 block 0
# |-----|--------------------------------------|--------------------------------------|
# | ... | a_carry_out_1 a_digit_1 a_carry_in_1 | a_carry_out_0 a_digit_0 a_carry_in_0 | op  A +
# | ... | b_carry_out_1 b_digit_1 b_carry_in_1 | b_carry_out_0 b_digit_0 b_carry_in_0 | op  B =
# |-----|--------------------------------------|--------------------------------------|--------
# | ... | s_carry_out_1 s_digit_1 s_carry_in_1 | s_carry_out_0 s_digit_0 s_carry_in_0 | sum S
# |-----|--------------------------------------|--------------------------------------|
#
# block schematic goal example:
# XOR a_digit_0, b_digit_0 -> C
# XOR C, a_carry_in_0 -> s_digit_0
# AND a_digit_0, b_digit_0 -> D
# AND C, a_carry_in_0 -> E
# OR E, D -> a_carry_out_0
#
# cioè:
# s_digit_0 := ((a_digit_0 XOR b_digit_0) XOR a_carry_in_0)
# a_carry_out_0 := ((a_digit_0 XOR b_digit_0) AND a_carry_in_0) OR (a_digit_0 AND b_digit_0)
#
# e viene generato un grafo del circuito con graphviz
# https://stackoverflow.com/questions/7922960/block-diagram-layout-with-dot-graphviz
#
class AdderState(State):
    __slots__ = ['digits', 'hash']

    def __init__(self, digits: np.ndarray):
        self.digits: np.ndarray = digits
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.digits.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.digits, other.digits)


class AdderCircuit(Environment):
    moves: List[str] = ['NOT', 'AND', 'OR', 'XOR']
    moves_inputs_nr: List[int] = [1, 2, 2, 2]
    moves_rev: List[str] = ['NOT', 'NAND', 'NOR', 'XNOR']
    pieces = {1: '1', 0: '0'}
    batch_number = 0

    def __init__(self, dim: int):
        super().__init__()

        self.dim: int = dim
        self.dtype = np.uint8

        # Solved state
        self.goal_digits: np.ndarray = np.zeros((self.dim * 3 * 3,), dtype=self.dtype)

    def next_state(self, states: List[NMPuzzleState], action: int) -> Tuple[List[AdderState], List[float]]:
        # initialize
        # TODO
        pass

    def prev_state(self, states: List[NMPuzzleState], action: int) -> List[NMPuzzleState]:
        # TODO
        pass

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[AdderState], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_tiles.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[AdderState] = [AdderState(self.goal_tiles.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[AdderState]) -> np.ndarray:
        # TODO
        pass

    def state_to_nnet_input(self, states: List[AdderState]) -> List[np.ndarray]:
        # TODO
        pass

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        # TODO
        pass

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[AdderState],
                                                                                          List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            # idxs: np.ndarray = np.where(moves_lt)[0]
            # subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            # idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            # states_np[idxs], z_idxs[idxs], _ = self._move_np(states_np[idxs], z_idxs[idxs], move)

            # num_back_moves[idxs] = num_back_moves[idxs] + 1
            # moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[AdderState] = [AdderState(x) for x in list(states_np)]

        return states, scramble_nums.tolist()
        # TODO

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"
        # TODO
        pass

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
    def adder_block_table(a, b, cin) -> np.ndarray:  # SUM + carry out
        transition_table: Dict[str, np.ndarray] = {'000': np.array([0, 0]),
                                                   '001': np.array([1, 0]),
                                                   '010': np.array([1, 0]),
                                                   '011': np.array([0, 1]),
                                                   '100': np.array([1, 0]),
                                                   '101': np.array([0, 1]),
                                                   '110': np.array([0, 1]),
                                                   '111': np.array([1, 1])
                                                   }

    @staticmethod
    def load_state(state_2d):
        return np.array(state_2d, dtype=np.int).flatten()

    @staticmethod
    def entry_user_state() -> List[State]:
        # example
        # [[ 0,  0,  0],
        #  [ 0,  0,  0],
        #  [ 0,  0,  0]]
        entry = input('entry board as [[...],[...],...] blocks: ')
        init_state = ast.literal_eval(entry)
        return [AdderState(AdderCircuit.load_state(init_state))]

    def render(blocks: np.ndarray, n: int):
        bblocks = np.reshape(board, (n, 3, 3))
        s_01 = ''
        s_02 = ''
        s_03 = ''
        s_04 = ''
        for b in range(n - 1, -1, -1):
            s_01 += '|----------|'
            s_02 += '| {} {} {} |'.format(bblocks[n][0][0], bblocks[n][0][1], bblocks[n][0][2])
            s_03 += '| {} {} {} |'.format(bblocks[n][1][0], bblocks[n][1][1], bblocks[n][1][2])
            s_04 += '| {} {} {} |'.format(bblocks[n][2][0], bblocks[n][2][1], bblocks[n][2][2])
        logger_main.info(s_01)
        logger_main.info(s_02)
        logger_main.info(s_03)
        logger_main.info(s_04)
