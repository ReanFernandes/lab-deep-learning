import os
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_data(log_dir, tags, output_dir='/home/rean/DL_lab_submission_2/'):
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    plt.figure(figsize=(10, 10))
    plt.grid()
    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        plt.plot(steps, values, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.legend()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,  'mIoU_shared_qk.png')
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

log_dir = 'outputs/train_segmentation/SharedQK/2023_05_16_18_38_30'

tags = ["validation/mIoU"]

plot_tensorboard_data(log_dir, tags)