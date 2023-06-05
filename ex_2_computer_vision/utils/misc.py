import numpy as np
import matplotlib.pyplot as plt
import datetime
from typing import Optional


def get_timestamp_for_filename(dtime: Optional[datetime.datetime] = None):
    """
    Convert datetime to timestamp for filenames.

    Args:
        dtime: Optional datetime object, will use now() if not given.

    Returns:
        timestamp as string
    """
    if dtime is None:
        dtime = datetime.datetime.now()
    ts = str(dtime).split(".", maxsplit=1)[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts


class SegmentationDisplay:
    def __init__(self, classes, overlay_mix=0.65, ignore_zero_idx=True):
        """
        classes: a list of strings containing the dataset's class names
        overlay_mix: the proportion of segmentation mask in the mix with the image
        """
        assert 0<=overlay_mix<=1
        self.ignore_zero_idx = ignore_zero_idx
        self.classes = classes if not ignore_zero_idx else classes[1:]
        self.overlay_mix = overlay_mix
    
    def draw_and_save(self, pil_image, torch_segm, dest):
        """
        pil_image: 
        """
        torch_segm = torch_segm.squeeze()
        assert len(torch_segm.shape) == 2
        segm_np = torch_segm.numpy()
        mask = np.logical_and(segm_np>0, segm_np<255)[...,None]*self.overlay_mix
        mix = (plt.cm.tab20(segm_np, bytes=True)[...,:3]*mask+np.array(pil_image)*(1-mask)).astype(np.uint8).astype(np.uint8)

        # legend
        colors = [plt.cm.tab20(i) for i in range(int(self.ignore_zero_idx), len(self.classes))]
        f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
        handles = [f("s", c) for c in colors]
        legend = plt.legend(handles, self.classes, ncols=1, bbox_to_anchor=(0, 1), framealpha=1, frameon=False, fontsize=8)

        plt.axis('off')
        plt.tight_layout()
        plt.imshow(mix)
        plt.savefig(dest, bbox_extra_artists=(legend,), bbox_inches='tight')


