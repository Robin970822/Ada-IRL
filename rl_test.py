"""
Created on Tue April 24 21:07 2018

@author: hanxy
"""
from maze_env import Maze
from rl import *

RL_EPISODE = 10


def q_update():
    for episode in range(RL_EPISODE):
        print "Episode %d" % episode
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
    env.destroy()


def sarsa_update():
    for episode in range(RL_EPISODE):
        print "Episode %d" %(episode)
        # initial observation
        observation = env.reset()
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # choose action
            action_ = RL.choose_action(str(observation))

            # make action
            observation_, reward, done = env.step(action)

            # learn
            RL.learn(str(observation), action, reward,
                     str(observation_), action_)

            # update observation
            observation = observation_
            action = action_

            if done:
                break
    print 'Game Over'
    print RL.q_table
    env.destroy()


def sarsa_lambda_update():
    for episode in range(RL_EPISODE):
        print "Episode %d" %(episode)
        # initial observation
        observation = env.reset()
        action = RL.choose_action(str(observation))
        RL.eligibility_trace *= 0
        while True:
            # fresh env
            env.render()

            # choose action
            action_ = RL.choose_action(str(observation))

            # make action
            observation_, reward, done = env.step(action)

            # learn
            RL.learn(str(observation), action, reward,
                     str(observation_), action_)

            # update observation
            observation = observation_
            action = action_

            if done:
                break
    print 'Game Over'
    print RL.q_table
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)))
    env.after(100, q_update)
    env.mainloop()

    env = Maze()
    RL = Sarsa(actions=list(range(env.n_actions)))
    env.after(100, sarsa_update)
    env.mainloop()

    env = Maze()
    RL = SarsaLambda(actions=list(range(env.n_actions)))
    env.after(100, sarsa_lambda_update)
    env.mainloop()

