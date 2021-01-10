#
# draw board on console
#
from typing import List
import numpy as np
from .loggers import logger_main

pieces = None

SEPARATOR_LEN = 40
moves: List[str] = ['UP ^^^', 'DOWN vvv', '<-- LEFT', 'RIGHT -->']
batch_number = 0


def load_pieces(size_rows: int, size_cols: int):
    global pieces

    pieces = {}
    for i in range(1, size_cols * size_rows):
        pieces[str(i)] = "{:2d}".format(i)
    pieces[str(0)] = "  "


def render(board: np.ndarray, size_rows: int, size_cols: int):
    if pieces is None:
        load_pieces(size_rows, size_cols)

    bboard = np.reshape(board, (size_rows, size_cols))
    for r in range(size_rows):
        s = str([pieces[str(x)] for x in bboard[r]])
        logger_main.info([pieces[str(x)] for x in bboard[r]])


def render_path(board_path: np.ndarray, soln):
    global batch_number

    batch_number += 1
    i = 0
    first = True
    for item in board_path:
        if first:
            first = False
            logger_main.info('-' * SEPARATOR_LEN + ' BATCH NR: ' + str(batch_number))
        else:
            logger_main.info('-' * SEPARATOR_LEN + ' ' + str(i) + ' (' + moves[soln[i - 1]] + ')')
        render(item.tiles, size_rows=4, size_cols=4)
        i += 1
    logger_main.info('=' * SEPARATOR_LEN)
