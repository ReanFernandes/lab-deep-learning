
import os
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_tensorboard_data(log_dir, tags, output_dir='/home/rean/DL_lab_submission_1'):
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    plt.figure(figsize=(10, 6))

    for tag in tags:
        if tag in accumulator.Tags()['scalars']:
            events = accumulator.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            plt.hist(values, bins = 250,label=tag)
            plt.xlabel("Probability")
            plt.title("Probability Distribution of action : " + str(tag) )
            # plt.plot(steps, values, label=tag)
        else:
            print(f"Tag '{tag}' not found in log directory.")
    
    

    
    plt.ylabel('Frequency')
    plt.legend()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, str(tag) + '.png')
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


# Usage example:
log_dir = 'tensorboard/train_car_dqn/DQN_Agent_carracing-20230503-072035'  # Replace with your TensorBoard log directory
# tags = [ "straight", "left", "right", "accel", "brake"]
# tags = ["episode_reward"]
# tags = ["straight"]
# tags = ["left"]
# tags = ["right"]
# tags = ["accel"]
tags = ["brake"]

plot_tensorboard_data(log_dir, tags)