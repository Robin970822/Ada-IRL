"""
Created on Wed April 25 16:06 2018

@author: hanxy
"""
from maze_env import Maze
from rl import *
from irl import *
import time


class IRLTest:
    def __init__(self, episode=10, env=Maze, irl=SimpleAdaIRL, rl=QLearning):
        self.env = env()
        self.RL = rl(actions=list(range(self.env.n_actions)))
        self.IRL = irl(state_space=self.env.row * self.env.col)
        self.expert = np.array([0, 0])
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
            print "Episode %d" % episode

            # initial observation
            observation = self.env.reset()
            states = observation
            while True:
                # fresh env
                self.env.render()

                action = self.RL.choose_action(str(observation))

                observation_, reward, done = self.env.step(action)

                reward += self.IRL.get_reward(str(observation_))

                self.RL.learn(str(observation), action, reward,
                              str(observation_))

                observation = observation_
                states = np.vstack((states, observation))

                if done:
                    break
                # print states
            self.IRL.learn(states)

        # print 'Game Over'
        # print RL.q_table
        print self.IRL.reward_weight
        # env.destroy()

    def main(self):
        print 'Game begin'
        self.env.after(100, self.store_expert)
        self.env.mainloop()


if __name__ == '__main__':
    iri_test = IRLTest(episode=130)
    iri_test.main()
