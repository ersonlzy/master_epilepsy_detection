from engines import EngineBase
from alive_progress import alive_bar, alive_it
from itertools import cycle
import torch
import math
from utils import *
import threading
import swanlab
from tqdm import tqdm


class Engine4Exp(EngineBase):
    project = "master_epilepsy_detection"
    def __init__(self, args):
        super().__init__(args)
        self.expReady()
    

    @torch.no_grad()
    def run(self):
        self.model.eval()
        for samples, targets in tqdm(self.valid_dataloader):
            outputs = self.model.kernel(samples, targets)
            fig, ax = plt.subplots(figsize=(24, 2)) 
            sns.heatmap(outputs["feats"][0].cpu().numpy(), ax=ax, vmax=1, vmin=-1)
            ax.set_title(self.dataset.classes_list[targets[0].item()])
            swanlab.log({"feats": swanlab.Image(fig, mode="RGB"), 
                         })