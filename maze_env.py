"""
@author: hanxy
"""

import numpy as np
import tkinter as tk
import time

UNIT = 40  # pixels
SQUAR = UNIT * 3 / 8


# create points on canvas
def create_points(points, canvas, fill):
    points_center = UNIT / 2 + points * UNIT
    shape = np.shape(points_center)
    assert shape[-1] == 2, 'only build 2d maze'
    if len(shape) == 1:
        rect = canvas.create_rectangle(
            points_center[0] - SQUAR, points_center[1] - SQUAR,
            points_center[0] + SQUAR, points_center[1] + SQUAR,
            fill=fill
        )
    return rect


class Maze(tk.Tk, object):
    def __init__(self, col=8, row=8,
                 hell=np.array([[6, 1], [1, 6], [5, 6], [6, 5]]),
                 origin=np.array([0, 0]),
                 terminal=np.array([7, 7])):
        super(Maze, self).__init__()
        self.action_space = ['w', 's', 'a', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        # col and row for maze
        self.col = np.max([col, np.max(hell, axis=0)[1] + 1,
                           origin[1] + 1, terminal[0] + 1])
        self.row = np.max([row, np.max(hell, axis=0)[0] + 1,
                           origin[0] + 1, terminal[0] + 1])

        self.origin = origin  # origin coordinate
        self.terminal = terminal  # terminal coordinate
        self.hell = hell  # hell coordinate
        self.n_hells = np.shape(self.hell)[0]  # number of hells
        self.geometry('{0}x{1}'.format(self.row * UNIT,
                                       self.col * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.row * UNIT,
                                width=self.col * UNIT)

        # create grids
        for c in range(0, self.col * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.row * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.row * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.col * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin (red)
        self.rect = create_points(self.origin, self.canvas, 'red')

        # create terminal (yellow)
        self.terminal_point = create_points(self.terminal, self.canvas, 'yellow')

        # create hell (black)
        self.hell_points = np.zeros(self.n_hells, dtype=int)
        for i in range(0, self.n_hells):
            self.hell_points[i] = create_points(self.hell[i], self.canvas, 'black')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        self.rect = create_points(self.origin, self.canvas, 'red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up 'w'
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down 's'
            if s[1] < (self.row - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left 'a'
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right 'd'
            if s[0] < (self.col - 1) * UNIT:
                base_action[0] += UNIT

        # move agent
        self.canvas.move(self.rect, base_action[0], base_action[1])

        # next state
        s_ = self.canvas.coords(self.rect)

        # reward function for Reinforcement Learning
        if s_ == self.canvas.coords(self.terminal_point):
            reward = 100
            done = True
            s_ = 'terminal'
        else:
            hell_rects = np.zeros((self.n_hells, 4))
            for i in range(0, self.n_hells):
                hell_rects[i] = self.canvas.coords(self.hell_points[i])
            if s_ in hell_rects:
                reward = -10
                done = True
                s_ = 'terminal'
            else:
                reward = -1
                done = False
        return s_, reward, done

    def render(self):
        time.sleep(0.01)
        self.update()


if __name__ == '__main__':
    env = Maze()
    env.mainloop()
