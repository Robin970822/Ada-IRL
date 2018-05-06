"""
Created on Wed April 25 15:15 2018

@author: hanxy
"""

import numpy as np
import pandas as pd


def one_hot(indices, depth):
    return (np.arange(depth) == indices).astype(np.int32)


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
        # print len(states)
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
                pd.Series(1.0 / self.state_space, name=state)
            )

            self.reward_weight = self.reward_weight.append(
                pd.Series(0, name=state)
            )

    def reward(self, state):
        self.check_state_exist(state)
        return np.array(self.reward_weight.loc[state])[0]


class AdaIRL(object):
    def __init__(self, state_space, discount_factor=0.7):
        self.state_space = state_space
        self.ada_weight = np.ones(state_space) / state_space
        self.reward_weight = np.zeros(state_space)
        self.GAMMA = discount_factor

    def store_expert(self, expert):
        # step x state_code
        self.expert = expert
        # state_space x feature_dim
        self.expert_expectation = self._expectation(expert)

    def learn(self, states):
        # make states and expert in the same length
        if len(states) >= len(self.expert):
            states = states[0:len(self.expert)]
        else:
            states = np.append(states, -np.ones(len(self.expert) - len(states)).astype(np.int32))
        # print 'states', states

        # calculate searching direction
        equal, unequal, searching_direction = self._searching_direction(states)
        # print 'unequal', unequal
        # print 'searching_direction', searching_direction

        # calculate error rate
        error_rate = self._error_rate(states, unequal)
        # print 'error_rate', error_rate

        # update Ada weight
        self._update_ada_weight(error_rate, equal, unequal)
        # print 'ada_weight', self.ada_weight

        # update reward weight
        self._update_reward_weight(searching_direction)

    def reward(self, state):
        return self.reward_weight[state]

    def _error_rate(self, states, unequal):
        # print np.where(states_expection != self.expert_expectation)
        return np.sum(self.ada_weight[unequal])

    def _searching_direction(self, states):
        states_expectation = self._expectation(states)
        # print 'states_expectation', states_expectation
        equal = np.where(states_expectation == self.expert_expectation)
        unequal = np.where(states_expectation != self.expert_expectation)
        searching_direction = self.expert_expectation - states_expectation
        return equal, unequal, searching_direction

    def _update_ada_weight(self, error_rate, equal, unequal):
        self.ada_weight[equal] *= error_rate / (1 - error_rate)
        self.ada_weight[unequal] *= 1
        self.ada_weight /= np.sum(self.ada_weight)

    def _update_reward_weight(self, searching_direction):
        self.reward_weight += self.ada_weight * searching_direction
        # print 'reward weight', self.reward_weight

    # return the feature vector of state
    def _feature(self, state, feature_function=one_hot):
        return feature_function(state, self.state_space)

    # return the expectation of states sequence
    def _expectation(self, states):
        length = len(states)
        expectation = np.zeros(self.state_space)
        for i in range(length):
            # print 'states[i]', states[i]
            feature = self._feature(states[i])
            # print 'feature', feature
            expectation += (self.GAMMA ** i) * feature
        return expectation


if __name__ == '__main__':
    ada_irl = AdaIRL(5 * 5)
    expert = np.array([0, 1, 6, 7, 2])
    states = np.array([0, 1, 2, 3, 4])
    ada_irl.store_expert(expert)
    # print 'expert', ada_irl.expert
    # print 'expert_expectation', ada_irl.expert_expectation
    # print 'ada_weight', ada_irl.ada_weight
    ada_irl.learn(states)
