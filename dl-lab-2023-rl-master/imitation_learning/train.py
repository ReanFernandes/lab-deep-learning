from __future__ import print_function

import sys
sys.path.append("../") 
import json
import pickle
import numpy as np
import os
import gzip
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import gym 
from utils import *
from agent.bc_agent import BCAgent
from torch.utils.tensorboard import SummaryWriter
from tensorboard_evaluation import Evaluation
from imitation_learning.test import run_episode

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb') 
    data = pickle.load(f)
    
    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid,history_length):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    # i am using an nn with 3d convolutions, which means my data will have dimensions : (batch_size, 1 (for grayscale), history_length, 96, 96)
    
    X_train = np.array(rgb2gray(X_train))
    X_valid = np.array(rgb2gray(X_valid))
    y_train = np.array([action_to_id(a) for a in y_train])
    y_valid = np.array([action_to_id(a) for a in y_valid])

    #stack the frames based on history_length
    X_train = X_train.reshape((-1,history_length,96,96))
    X_valid = X_valid.reshape((-1,history_length,96,96))
    y_train = y_train[::history_length] # take only the last truth label in the history stack for training if h >1. Else it gives the whole y_train
    y_valid = y_valid[::history_length]

    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid,y_valid, num_epochs, batch_size, lr,best_reward, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")
    # create gym environement for testing
    env = gym.make('CarRacing-v0').unwrapped

    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    # agent = BCAgent(...)
    agent = BCAgent(history_length=history_length, output_classes=4, batch_size=batch_size, lr=lr)
    
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train_im_learn"),"DQN_Agent_carracing", ["Training loss", "Training accuracy", "Validation loss", "Validation accuracy"])
    y_train = torch.from_numpy(y_train)
    X_train = torch.from_numpy(X_train)
    
    # Create data loader with weighted random sampling for training data
    train_dataset = TensorDataset(X_train, y_train)
    class_counts = torch.bincount(y_train)
    class_weights = 1./class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    steps_per_epoch = len(train_loader)
    # create data loader for validation data with shuffling
    y_valid = torch.from_numpy(y_valid)
    X_valid = torch.from_numpy(X_valid)
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    
    # training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        for step,(X,y) in enumerate(train_loader): 
            
            train_loss, train_accuracy, valid_loss, valid_accuracy = 0, 0, 0, 0
            X = X.to(agent.device)
            y = y.to(agent.device)
            loss = agent.update(X, y)
            
            # calculate training acc/loss
            if step % steps_per_epoch//20 == 0 and step != 0:
                train_accuracy, train_loss = agent.get_accuracy_and_loss(X, y)
        
            # calculate validation acc/loss
            if step % steps_per_epoch//5  == 0 and step != 0:
                print("... validate model at step {}".format(step))
                valid_accuracy_buffer = 0
                valid_loss_buffer = 0
                for idx, (X_validation, y_validation) in enumerate(valid_loader):
                    X_validation = X_validation.to(agent.device)
                    y_validation = y_validation.to(agent.device)
                    valid_accuracy, valid_loss = agent.get_accuracy_and_loss(X_validation, y_validation)
                    valid_accuracy_buffer += valid_accuracy
                    valid_loss_buffer += valid_loss.item()
                valid_accuracy = valid_accuracy_buffer / len(valid_loader)
                valid_loss = valid_loss_buffer / len(valid_loader)
                tensorboard.write_episode_data(epoch * steps_per_epoch + step, {"Training loss": train_loss.item(), "Training accuracy": train_accuracy, "Validation loss": valid_loss, "Validation accuracy": valid_accuracy})
            
        # test trained model 
        print("... testing model at the end of epoch {}".format(epoch))
        current_reward = 0
        test_eps = 5 # collect rewards over these many episodes
        for i in range(test_eps):
            current_reward += run_episode(env, agent, rendering=False, max_timesteps=1000)
        current_reward /= test_eps

        if current_reward > best_reward:
            best_reward = current_reward
            agent.save(os.path.join(model_dir, "best_model_current_8.pt"))
            hyperparams = {"lr":lr,  "batch_size":batch_size, "num_epochs":num_epochs,  "history_length":history_length, "Best Reward": best_reward}
            #save hyperparams
            with open(os.path.join(model_dir, "best_model_params_8.txt"), "w") as f:
                json.dump(hyperparams, f)
            print("Current best model found at epoch {}, avg reward: {}".format(epoch, current_reward))
    return best_reward
    
if __name__ == "__main__":
    # read data    
    # X_train, y_train, X_valid, y_valid = read_data("./data")
    history_lengths = [ 8]
    batch_sizes = [ 128]
    num_epochs_s = [50]
    lr_s = [1e-4]
    best_reward = -9999
    # for some reason the code breaks after changing the history length. If i have the time, I will fix it. Currently i just erase whichever value is completed from the history length list
    for num_epochs in num_epochs_s:
        for lr in lr_s:
            for history_length in history_lengths:
                X_train, y_train, X_valid, y_valid = read_data("./data")
                X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid,history_length=history_length)
                for batch_size in batch_sizes:
                    # train model (you can change the parameters!)
                    best_reward = train_model(X_train, y_train, X_valid, y_valid, num_epochs=num_epochs, batch_size=batch_size, lr=lr, best_reward = best_reward)
                
