"""
Created on Wed April 25 15:15 2018

@author: hanxy
"""

import numpy as np
import pandas as pd


# Simple Ada-IRL implemented with Reward weight table and Ada-weight table
class SimpleAdaIRL(object):

    def __init__(self, state_space):
        self.state_space = state_space
        self.reward_weight = pd.DataFrame(dtype=np.float64)
        self.ada_weight = pd.DataFrame(dtype=np.float64)

    def store_expert(self, expert):
        self.expert = expert

    def learn(self, states):
        # make states and expert in the same length
        length = min(len(self.expert), len(states))
        states = states[0:length]
        expert = self.expert[0:length]
        print len(states)
        # calculate error_rate
        error_rate = 0
        for i in range(length):
            if (states[i] != expert[i]).any():
                # print states[i], expert[i]
                self.check_state_exist(str(states[i]))
                self.check_state_exist(str(expert[i]))
                error_rate += self.ada_weight.loc[str(states[i])] + self.ada_weight.loc[str(expert[i])]

        # print self.ada_weight
        # print self.reward_weight
        # print 'error_rate:', error_rate
        alpha = error_rate / (1 - error_rate)

        # update ada-weight table
        for i in range(length):
            if (states[i] == expert[i]).all():
                self.check_state_exist(str(states[i]))
                self.check_state_exist(str(expert[i]))
                self.ada_weight.loc[str(states[i])] *= alpha
                self.ada_weight.loc[str(expert[i])] *= alpha

        # normalize
        self.ada_weight /= (np.sum(self.ada_weight.iloc[:, 0]) +
                            (1.0 / self.state_space) * alpha * (self.state_space - len(self.ada_weight)))
        # print self.ada_weight

        # update reward weight table
        for i in range(length):
            if (states[i] != expert[i]).any():
                self.check_state_exist(str(states[i]))
                self.check_state_exist(str(expert[i]))
                self.reward_weight.loc[str(expert[i])] += self.ada_weight.loc[str(expert[i])]
                self.reward_weight.loc[str(states[i])] -= self.ada_weight.loc[str(states[i])]
        # print self.reward_weight

    def check_state_exist(self, state):
        if state not in self.ada_weight.index:
            # append new state to Ada-weight table
            self.ada_weight = self.ada_weight.append(
                pd.Series(1.0/self.state_space, name=state)
            )

            self.reward_weight = self.reward_weight.append(
                pd.Series(0, name=state)
            )

    def get_reward(self, state):
        self.check_state_exist(state)
        return np.array(self.reward_weight.loc[state])[0]


if __name__ == '__main__':
    ada_irl = SimpleAdaIRL(5*5)
    expert = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                       [4, 1], [4, 2], [4, 3], [4, 4]])
    states = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2],
                      [2, 3], [1, 3], [0, 3], [0, 4], [1, 4]])
    ada_irl.store_expert(expert)
    ada_irl.learn(states)
