#
# 10.1.2021
# interfaccia per la soluzione del puzzle N x M
#
from typing import List
from environments.environment_abstract import Environment, State
from utils import env_utils
from argparse import ArgumentParser
import pickle
import os
import time

from multiprocessing import Queue, Process


# distanza 24 esito: non risolta in 48h da IDA* base
init_state = (
    (1,   2, 11,  3,  4,  6, 16,  7),
    (10, 25, 13, 12,  5,  0, 14,  8),
    (9,  20, 18, 27, 22, 23, 15, 24),
    (17, 26, 19, 28, 21, 29, 30, 31),
)


def main():
    pass


if __name__ == "__main__":
    main()
