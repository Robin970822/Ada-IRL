"""
Created on Tue April 24 21:29 2018

@author: hanxy
"""

import numpy as np
import pandas as pd


# Base Reinforcement Learning
class RL(object):

    def __init__(self, actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9):
        self.actions = actions
        self.n_actions = len(actions)
        self.LR = learning_rate
        self.Gamma = reward_decay
        self.Epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions,
                                    dtype=np.float64)

    # choose action based on observation
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.Epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index
            ))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass

    # check if state exists
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    np.zeros(self.n_actions),
                    index=self.q_table.columns,
                    name=state
                )
            )
