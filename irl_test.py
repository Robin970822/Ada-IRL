"""
Created on Wed April 25 16:06 2018

@author: hanxy
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from maze_env import Maze
from rl import *
from irl import *
import time
import img_utils


class IRLTest:
    def __init__(self, episode=10, env=Maze, irl=AdaIRL, rl=QLearning):
        self.env = env(hell=np.array([[6, 1], [1, 6], [4, 7], [7, 4]]), terminal=np.array([5, 5]))
        self.RL = rl(actions=list(range(self.env.n_actions)))
        self.IRL = irl(state_space=self.env.row * self.env.col)
        self.expert = 0
        self.Episode = episode

    # store expert from human demonstration
    def store_expert(self):
        self.env.bind('<Key>', self.on_key_pressed)
        self.env.bind('<Key-space>', self.on_space_pressed)
        print 'In human control'
        print 'Press ', self.env.action_space, ' to control move'
        print 'Press space to store expert'

    # key pressed call back function
    def on_key_pressed(self, event):
        char = event.char
        self.env.render()
        if char in self.env.action_space:
            action = self.env.action_space.index(char)
            s, r, done = self.env.step(action)
            self.expert = np.vstack((self.expert, s))
            if done:
                self.env.reset()

    # space pressed call back function
    def on_space_pressed(self, event):
        print 'Storing expert...'
        self.IRL.store_expert(self.expert)
        print 'expert: ', self.expert
        self.env.unbind('<Key>')
        self.env.unbind('<Key-space>')
        print 'Out of human control'
        self.env.bind('<Key-Return>', self.on_enter_pressed)
        print 'Press Enter to start IRL process'

    # Enter pressed call back function
    def on_enter_pressed(self, event):
        print 'IRL process will start in 100ms'
        time.sleep(0.1)
        self.ada_irl_update()

    # test Ada-IRL in simple maze
    def ada_irl_update(self):
        for episode in range(self.Episode):
            # initial observation
            observation = self.env.reset()
            # action = self.RL.choose_action(observation)
            states = observation
            eq_r = 0
            eq_step = 0
            while True and eq_step < len(self.IRL.expert)*3:
                # print eq_step
                # fresh env
                self.env.render()

                action = self.RL.choose_action(observation)

                observation_, reward, done = self.env.step(action)

                reward = self.IRL.reward(observation_,)

                self.RL.learn(observation, action, reward,
                              observation_)

                observation = observation_
                # action = action_
                states = np.vstack((states, observation))
                eq_r += reward
                eq_step += 1

                if done:
                    break
                # print states
            self.IRL.learn(states)
            print "Episode %d | Reward" % episode, eq_r
            # print self.IRL.reward_weight

        print 'Game Over'
        print self.RL.q_table
        reward_weight = self.IRL.reward_weight.reshape([self.env.col, self.env.row])
        expert_reward = self.IRL.reward_weight[self.expert]
        print reward_weight

        actual_reward = np.zeros_like(reward_weight)
        actual_reward[5, 5] = 10
        actual_reward[6, 1] = -5
        actual_reward[1, 6] = -5
        actual_reward[4, 7] = -5
        actual_reward[7, 4] = -5
        print actual_reward

        plt.figure(figsize=(25, 10))
        plt.subplot(1, 2, 1)
        img_utils.heatmap2d(actual_reward, 'Reward MAP - Ground Truth', block=False)
        plt.subplot(1, 2, 2)
        img_utils.heatmap2d(reward_weight, 'Reward MAP - ddlGAN', block=False)
        plt.show()

        img_utils.heatmap3d(reward_weight, 'Reward MAP - ddlGAN')
        plt.show()

        print expert_reward
        x = np.arange(len(self.IRL.expert))
        plt.plot(x, expert_reward, 'r-', lw=5)
        plt.show()

    def main(self):
        print 'Game begin'
        self.env.after(100, self.store_expert)
        self.env.mainloop()


if __name__ == '__main__':
    iri_test = IRLTest(episode=300, rl=QLearning)
    iri_test.main()
