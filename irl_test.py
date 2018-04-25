"""
Created on Wed April 25 16:06 2018

@author: hanxy
"""
from maze_env import Maze
from rl import *
from irl import *

IRL_EPISODE = 100


def ada_irl_update():
    for episode in range(IRL_EPISODE):
        print "Episode %d" % episode

        # initial observation
        observation = env.reset()
        states = observation
        while True:
            # fresh env
            env.render()

            action = RL.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            reward += IRL.get_reward(str(observation_))

            RL.learn(str(observation), action, reward,
                     str(observation_))

            observation = observation_
            states = np.vstack((states, observation))

            if done:
                break
            # print states
        IRL.learn(states)

    # print 'Game Over'
    # print RL.q_table
    print IRL.reward_weight
    # env.destroy()


if __name__ == '__main__':
    env = Maze()
    IRL = SimpleAdaIRL(state_space=env.row*env.col)
    RL = QLearning(actions=list(range(env.n_actions)))
    expert = np.array([[0, 0], [1, 0], [1, 1], [2, 1],
                      [2, 2], [3, 2], [4, 2], [5, 2],
                      [6, 2], [6, 3], [7, 3], [7, 4],
                      [7, 5], [7, 6], [7, 7]])
    IRL.store_expert(expert)
    env.after(100, ada_irl_update)
    env.mainloop()
