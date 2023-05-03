from __future__ import print_function

import sys
sys.path.append("../") 
from gym.wrappers import Monitor

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    # start of the episode, I will create a queue of entries according to history. The stack will have the first h-1 frames as blank screen, and 
    # remaining states will be appended to the queue as they come.
    state = rgb2gray(state)
    blank_frame = np.zeros_like(state)
    #concatenate along new axis to create a stack of frames
    state_array = np.concatenate([blank_frame[np.newaxis, ...] for _ in range(agent.history_length-1)] + [state[np.newaxis, ...]], axis=0)
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...
        state = state_array.reshape((1,agent.history_length  , 96, 96))
        a = agent._get_predictions(agent.predict(state))
        a = id_to_action(a)
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        # append the next state to the state_queue
        state_array= np.concatenate([state_array[1:], rgb2gray(next_state)[np.newaxis,...]], axis=0)
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True
    n_test_episodes = 15                # number of episodes to test

    # TODO: load agent
    # agent = BCAgent(...)
    # agent.load("models/bc_agent.pt")

    hyperparams = json.load(open("models/best_model_params_8.txt"))
    history_length = hyperparams["history_length"]
    batch_size = hyperparams["batch_size"]
    lr = hyperparams["lr"]
    

    
    agent = BCAgent(history_length=history_length, batch_size=batch_size, lr=lr)
    agent.load("models/best_model_current_8.pt")
    env = gym.make('CarRacing-v0').unwrapped
    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    print("mean: ", results["mean"])
    print("std: ", results["std"])


    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
