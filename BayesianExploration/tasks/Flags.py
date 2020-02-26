import numpy as np

from tasks.Task import Task

LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3


class Flags(Task):

    def __init__(self, maze, initial):
        super().__init__()
        self.initial = (initial[0], initial[1], 0)
        self.maze = maze
        self.height, self.width = maze.shape

        # count all the sub-goals and free cells
        self.goals = 0
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] > 0:
                    self.goals = max(self.goals, maze[r, c])

    def initial_state(self, training=True):
        return self.initial

    def valid_actions(self):
        return 4

    def check(self, row, col, action):
        if (row == 0 and action == UP) or \
            (row == self.height - 1 and action == DOWN) or \
            (col == 0 and action == LEFT) or \
            (col == self.width - 1 and action == RIGHT):
            return False
        return True

    def transition(self, state, action):

        # perform the movement
        row, col, collected = state
        if self.check(row, col, action):
            if action == LEFT:    col -= 1
            elif action == UP:    row -= 1
            elif action == RIGHT: col += 1
            elif action == DOWN:  row += 1
        else:
            return (row, col, collected), -0.2, False

        # compute the new state, status and reward
        grid_x1y1 = self.maze[row, col]
        if grid_x1y1 > 0:
            if collected + 1 == grid_x1y1:
                if collected + 1 == self.goals:
                    return (row, col, collected + 1), -0.1, True
                else:
                    return (row, col, collected + 1), -0.1, False
            elif collected >= grid_x1y1:
                return (row, col, collected), -0.1, False
            else:
                return (state[0], state[1], collected), -0.2, False
        else:
            return (row, col, collected), -0.1, False

    def default_encoding(self, state):
        h, w = self.height, self.width
        arr = np.zeros(h + w + 1 + self.goals, dtype=np.float32)
        arr[state[0]] = 1.0
        arr[h + state[1]] = 1.0
        arr[h + w + state[2]] = 1.0
        arr = arr.reshape((1, -1))
        return arr
