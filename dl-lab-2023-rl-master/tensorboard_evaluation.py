import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Evaluation:

    def __init__(self, store_dir, name, stats=[]):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        self.folder_id = "%s-%s" % (name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = SummaryWriter(os.path.join(store_dir, self.folder_id))
        self.stats = stats

    def write_episode_data(self, episode, eval_dict):
        """
         Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
         e.g. eval_dict = {"loss" : 1e-4}
        """

        for k in eval_dict:
            assert (k in self.stats)
            self.summary_writer.add_scalar(k, eval_dict[k], global_step=episode)

        self.summary_writer.flush()

    def close_session(self):
        self.summary_writer.close()