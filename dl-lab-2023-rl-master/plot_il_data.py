# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# # Loading event file
# event_acc = EventAccumulator('tensorboard/train_im_learn/DQN_Agent_carracing-20230503-061411')
# event_acc.Reload()

# # Show all tags in the log file
# print(event_acc.Tags())

# # Define the tag(s) to plot
# tags = ["Training loss", "Training accuracy", "Validation loss", "Validation accuracy"]

# # Extract data
# scalar_data = {}
# for tag in tags:
#     scalar_data[tag] = []
#     for scalar_event in event_acc.Tags()['scalars']:
#         if scalar_event in event_acc.Scalars(tag):
#             scalar_data[tag].append([scalar_event.step, scalar_event.value])

# # Plot data
# fig, ax = plt.subplots(figsize=(12,8))
# for tag in tags:
#     data = scalar_data[tag]
#     steps = [row[0] for row in data]
#     values = [row[1] for row in data]
#     ax.plot(steps, values, label=tag)

# # Set axis labels
# ax.set_xlabel('Steps')
# ax.set_ylabel('Value')

# # Add legend and show plot
# ax.legend()
# plt.savefig('im_plot.png')
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
            plt.plot(steps, values, label=tag)
        else:
            print(f"Tag '{tag}' not found in log directory.")
    plt.title('Accuracy for history_length = 8')
    # plt.title('Loss for history_length = 8')

    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.legend()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'accu_8.png')
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


# Usage example:
log_dir = 'tensorboard/train_im_learn/DQN_Agent_carracing-20230503-085429'  # Replace with your TensorBoard log directory
tags = [ "Training accuracy",  "Validation accuracy"] # Replace with your desired tags
# tags = ["Training loss", "Validation loss"] # Replace with your desired tags

plot_tensorboard_data(log_dir, tags)