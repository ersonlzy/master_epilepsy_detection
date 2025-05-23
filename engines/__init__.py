import torch
from models import *
import os
import swanlab
import torch
from models import registered_models, registered_optimizers, registered_schedulers
from data import registered_datasets
from copy import deepcopy
from utils import Metrics

from engines.engine4Cla import Engine4Cla

class EngineBase:
    project = "base"
    def __init__(self, args):
        self.args = args
        self.current_args = args
        self.experiemnt_name = args.experiment_name if args.experiment else f"{self.project}_{args.model}"
        self.createExp()
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.swanlab = swanlab.init(
            project=self.project,
            experiment_name=self.experiemnt_name,
            config={
                "model": args.model,
                "optim": args.optim,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "device": args.device,
            },
        )
        self.model = registered_models[args.model](args)
        self.dataset = registered_datasets[args.dataset](args)
        self.optimizer = registered_optimizers[self.args.optim](self.model.parameters(), 
                                                                self.args.lr, 
                                                                [0,9, 0,999], 
                                                                weight_decay=self.args.weight_decay)
        
        self.lr_scheduler = registered_schedulers[self.args.scheduler](self.optimizer, T_0=self.args.t0,
                                                                       T_mult=self.args.tmult, 
                                                                       eta_min=self.args.lr0, 
                                                                       last_epoch=self.args.init_epoch)
        self.metrics = Metrics(args)
        
    
    def expReady(self):
        self.train_dataloader = deepcopy(self.dataset.update("train"))
        self.valid_dataloader = deepcopy(self.dataset.update("valid"))
        self.mode.to(self.device)

    def saveExp(self):
        path = os.path.join(self.exp_path, "checkpoints")
        if not os.path.exists(path):
            os.mkdir(path)
        checkpoint_path = os.path.join(path, "checkpoint{self.current_args.epoch}{self.current_args.step}")
        os.mkdir(checkpoint_path)
        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        current_args = self.current_args.__dict__
        args = self.args.__dict__
        torch.save(model_state_dict, os.path.join(checkpoint_path, "model.pth"))
        torch.save(optim_state_dict, os.path.join(checkpoint_path, "optim.pth"))
        with open(os.path.join(checkpoint_path, "current_args.yaml"), "w") as f:
            for arg, value in current_args.items():
                f.writelines(f"{arg} : {value} \n")
        with open(os.path.join(checkpoint_path, "args.yaml"), "w") as f:
            for arg, value in args.items():
                f.writelines(f"{arg} : {value} \n")

        

        

    def createExp(self):
        if not os.path.exists("./runs"):
            os.mkdir("./runs")
        i = 0
        for folder in os.listdir("./runs"):
            if self.experiemnt_name in folder:
                i += 1
                continue
        self.exp_path = f"./runs/{self.experiemnt_name}{i}"
        os.mkdir(self.exp_path)


        

registered_engines = {
    "base": EngineBase,
    "cla": Engine4Cla,
}



registered_optimizers = {
    "adam": torch.optim.AdamW,
}

registered_schedulers = {
    "coswr": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(),
}


