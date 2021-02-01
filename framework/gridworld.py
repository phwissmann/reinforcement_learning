
from enum import Enum
import numpy as np


class Action(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class Grid():

    def __init__(self, rows: int, cols: int):

        self.rows = rows
        self.cols = cols
        self.special_state_A = 1
        self.special_state_A_successor = 1 + 4*self.cols
        self.special_state_B = 3
        self.special_state_B_successor = 1 + 2*self.cols

    def successor_state(self, state: int, action: Action):
        # if special state A, all actions go to A's successor
        if state == self.special_state_A:
            return self.special_state_A_successor

        if state == self.special_state_B:
            return self.special_state_B_successor

        # else normal rules apply
        if action == Action.NORTH:
            return state - self.cols if state - self.cols > 0 else state
        elif action == Action.SOUTH:
            return state + self.cols if state + self.cols < self.rows*self.cols else state
        elif action == Action.EAST:
            return state + 1 if (state+1) % self.cols != 0 else state
        elif action == Action.WEST:
            return state - 1 if state % self.cols != 0 else state

    def state_index(self, col: int, row: int):
        return col + self.cols*row

    def reward(self, state: int, successor_state: int):

        if state == self.special_state_A and successor_state == self.special_state_A_successor:
            return 10.0

        if state == self.special_state_B and successor_state == self.special_state_B_successor:
            return 5.0

        if successor_state == state:
            return -1.0
        else:
            return 0.0

    def prob_action_selection(self, action: Action, state: int):
        return 0.25

    def prob_action_executed(self, state: int, successor_state: int):
        if state == successor_state:
            return 1
        return 1


DISCOUNT_RATE = 0.9


def solve_gridworld(grid: Grid):
    dim = grid.rows*grid.cols
    A = np.zeros((dim, dim))
    b = np.zeros((dim,))

    # for each state, fill entries in matrix at corresponding location
    for i in range(dim):
        A[i, i] = 1.0
        for action in Action:
            successor_state = grid.successor_state(i, action)

            A[i, successor_state] = A[i, successor_state] - grid.prob_action_selection(
                i, successor_state) * grid.prob_action_executed(i, successor_state) * DISCOUNT_RATE

            b[i] = b[i] + grid.prob_action_selection(action, i) * grid.prob_action_executed(
                i, successor_state) * grid.reward(i, successor_state)

    # with np.printoptions(precision=3, suppress=True):
     #   print(A)
       # print(b)

    x = np.linalg.solve(A, b)

    with np.printoptions(precision=2, suppress=True):
        print(x.transpose())


if __name__ == "__main__":
    grid = Grid(5, 5)

    solve_gridworld(grid)
