"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable
import matplotlib.pyplot as plt

mlearning_rate = 0.01
mreward_decay = 0.9
me_greedy = 0.9

r_list = []
def update():
    for episode in range(1000):
        if episode % 100 == 0:
            print ("Training episode is: "+ str(episode))
        # initial observation
        observation = env.reset()
        total_reward = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            total_reward += reward

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
        r_list.append(total_reward)

    # end of game
    print('game over')
    RL.getQtable().to_csv("data.csv")

    env.destroy()

    plt.figure(figsize=(14, 7))
    plt.plot(range(len(r_list)), r_list)
    plt.xlabel('Games played')
    plt.ylabel('Reward')
    plt.show()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)), learning_rate=mlearning_rate,
                        reward_decay=mreward_decay, e_greedy=me_greedy)

    env.after(100, update)
    env.mainloop()