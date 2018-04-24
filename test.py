"""
Created on Tue April 24 21:07 2018

@author: hanxy
"""
from maze_env import Maze
from rl import *


def update():
    for episode in range(100):
        print "Episode %d" %(episode)
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # choose action
            action = RL.choose_action(str(observation))

            # make action
            observation_, reward, done = env.step(action)

            # learn
            RL.learn(str(observation), action, reward,
                     str(observation_))

            # update observation
            observation = observation_

            if done:
                break
    print 'Game Over'
    print RL.q_table
    # env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
