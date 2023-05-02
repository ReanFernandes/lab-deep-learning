from __future__ import print_function
from datetime import datetime
import gym
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *
import os
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  0

    #TODO: Define networks and load agent
    # ....
    num_eval_episodes = 5
    eval_cycle = 20
    history_length = 4
    batch_size = 64
    num_actions = 5
    gamma=0.95
    epsilon=0.1
    tau=0.01
    lr=1e-4
                
   
    Q = CNN( history_length=history_length, output_classes=num_actions, batch_size=batch_size)
    Q_target = CNN( history_length=history_length, output_classes=5, batch_size=batch_size)
    agent = DQNAgent(Q=Q, Q_target=Q_target, num_actions=num_actions, batch_size=batch_size, gamma=gamma, epsilon=epsilon, tau=tau, lr=lr)
    agent.load("./models_carracing/carracing_dqn_agent.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True, history_length=history_length)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

