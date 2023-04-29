from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

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


def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    # agent = BCAgent(...)
    agent = BCAgent(history_length=history_length, output_classes=4, batch_size=batch_size, lr=lr)
    stats = ["Training loss", "Training accuracy", "Validation loss", "Validation accuracy"]
    tensorboard_eval = Evaluation(tensorboard_dir, "Training", stats)
    #pregenerate batch indices for shuffling
    batch_idx_training = np.arange(X_train.shape[0])
    batch_idx_validation = np.arange(X_valid.shape[0])
    
    def sample_minibatch(X, y, batch_idx,batch_size):
        index_list = np.random.choice(batch_idx, batch_size, replace=False)
        return X[index_list], y[index_list]

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)
    # training loop
    for i in range(n_minibatches):
        # sample minibatch
        X_batch_train, y_batch_train = sample_minibatch(X_train, y_train, batch_idx_training, batch_size)
        
        # update
        loss = agent.update(X_batch_train, y_batch_train)
        print(i)
        if i % 25 == 0:
            # sample from validation batch and get validation monitors
            X_batch_valid, y_batch_valid = sample_minibatch(X_valid, y_valid, batch_idx_validation, batch_size)
            valid_accuracy, valid_loss = agent.get_accuracy_and_loss(X_batch_valid, y_batch_valid)
            
            #use preexisting x_batch_train and y_batch_train to get training monitors
            training_accuracy, training_loss = agent.get_accuracy_and_loss(X_batch_train, y_batch_train)

            # write to tensorboard
            eval_dict = {"Training loss": training_loss.item,
                         "Training accuracy": training_accuracy,
                         "Validation loss": valid_loss.data,
                         "Validation accuracy": valid_accuracy}
            print(eval_dict)
            # tensorboard_eval.write_episode_data(i, eval_dict)




    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "test_agent.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")
    history_length = 1
    batch_size = 10
    n_minibatches = X_train.shape[0] // batch_size
    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid,history_length=history_length)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=n_minibatches, batch_size=batch_size, lr=1e-4)
 
