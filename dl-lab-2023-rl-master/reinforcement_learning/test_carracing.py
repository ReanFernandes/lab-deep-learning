from __future__ import print_function
from datetime import datetime
import json
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
    history_length = 2 # somwething broke while training, i couldnt point out what it was. The model was
                        #trained on history 5, but while loading it, keeping it 5 here broke it. I have made this fix and
                        #it works for now. I made the number of channels match for both.This was the error:
                        #RuntimeError: Given groups=1, weight of size [32, 3, 8, 8], expected input[32, 6, 96, 96] to have 3 channels, but got 6 channels instead
    batch_size = 32
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

    fname = "./results/current_best_carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

