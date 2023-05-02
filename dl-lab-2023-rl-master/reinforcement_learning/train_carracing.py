# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray, action_to_id
from utils import *
from agent.dqn_agent import DQNAgent
from agent.networks import CNN


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(history_length + 1,96, 96 )
    i=0
    while True:
        i+=1
        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)
        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(history_length + 1,96, 96 )

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard",best_reward=-1000):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"),"DQN_Agent_carracing", ["episode_reward", "straight", "left", "right", "accel", "brake"])

    for i in range(num_episodes):
        print("episode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        
        # scheduler for max time steps
        if i < 100:
            max_timesteps = 250
        elif i < 200:
            max_timesteps = 500
        else:
            max_timesteps = 1000

        stats = run_episode(env, agent, max_timesteps=max_timesteps, skip_frames=5 ,deterministic=False, do_training=True,history_length=history_length)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        if i % eval_cycle == 0:
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=True, do_training=False, history_length=history_length)
                tensorboard.write_episode_data(i + j, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                                  "straight" : stats.get_action_usage(STRAIGHT),
                                                                  "left" : stats.get_action_usage(LEFT),
                                                                  "right" : stats.get_action_usage(RIGHT),
                                                                  "accel" : stats.get_action_usage(ACCELERATE),
                                                                  "brake" : stats.get_action_usage(BRAKE)
                                                                  })
        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            if stats.episode_reward > best_reward:
                best_reward = stats.episode_reward
                agent.save(os.path.join(model_dir, "current_best_carracing_dqn_agent.pt"))
            agent.save(os.path.join(model_dir, "carracing_dqn_agent.pt"))

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    history_length = 4
    batch_size = 64
    num_actions = 5
    gamma=0.95
    epsilon=0.1
    tau=0.01
    lr=1e-4
    num_episodes = 1000
    current_reward = -999
                
    env = gym.make('CarRacing-v0').unwrapped
    
    # TODO: Define Q network, target network and DQN agent
    # ...
    Q = CNN( history_length=history_length, output_classes=num_actions, batch_size=batch_size)
    Q_target = CNN( history_length=history_length, output_classes=5, batch_size=batch_size)
    agent = DQNAgent(Q=Q, Q_target=Q_target, num_actions=num_actions, batch_size=batch_size, gamma=gamma, epsilon=epsilon, tau=tau, lr=lr)
    train_online(env, agent, num_episodes=100, history_length=history_length, model_dir="./models_carracing", best_reward=current_reward)

