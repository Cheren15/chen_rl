import gym
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append('../contents/5_Deep_Q_Network')
from RL_brain import DeepQNetwork

def run_game():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_

            r1 = (env.x_threshold - abs(x))/env.x_threshold
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians
            reward = r1 + r2

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
    RL.plot_cost()

def env_test():
    print(env.observation_space, env.action_space)
    observation = env.reset()
    print(observation)

if __name__ == "__main__":
    # maze game
    env = gym.make('CartPole-v0')
    env_test()
    RL = DeepQNetwork(env.action_space.n, env.observation_space.shape[0],
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_game()
