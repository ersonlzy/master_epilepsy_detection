import torch
from models import *
import os
import swanlab
import torch
from models import *
from data import registered_datasets
from copy import deepcopy
from utils import Metrics
from models.model import DecompNet4ESD
import datetime
from torch.utils.data import (
    DataLoader,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
)


class EngineBase:
    project = "base"
    def __init__(self, args):
        self.args = args
        self.current_args = args
        self.experiemnt_name = args.experiment if args.experiment else f"{self.project}_{args.model}"
        self.experiemnt_name = f"{self.experiemnt_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}"
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
        self.optimizer = registered_optimizers[self.args.optim](self.model.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        total_step = len(self.dataset) // self.args.batch_size * self.args.num_epochs
        self.lr_scheduler = registered_schedulers[self.args.scheduler](self.optimizer, T_max=total_step, eta_min=self.args.lr0, last_epoch=self.args.init_epoch - 1)
        self.metrics = Metrics(args)

    def expReady(self):
        self.dataset.update("train")
        train_dataset = deepcopy(self.dataset)
        self.dataset.update("valid")
        valid_dataset = deepcopy(self.dataset)
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        train_batch_sampler = BatchSampler(train_sampler, self.args.batch_size, drop_last=True)
        valid_batch_sampler = BatchSampler(valid_sampler, self.args.batch_size, drop_last=False)
        self.train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=self.dataset.collateFn4Cla, num_workers=0)
        self.valid_dataloader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler, collate_fn=self.dataset.collateFn4Cla, num_workers=0)
        self.model = self.model.to(self.device)

    def saveExp(self):
        path = os.path.join(self.exp_path, "checkpoints")
        if not os.path.exists(path):
            os.mkdir(path)
        checkpoint_path = os.path.join(path, f"checkpoint-{self.current_args.step}")
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
        self.exp_path = f"./runs/{self.experiemnt_name}"
        os.mkdir(self.exp_path)





registered_optimizers = {
    "adamw": torch.optim.AdamW,
}

registered_schedulers = {
    "coslr": torch.optim.lr_scheduler.CosineAnnealingLR,
}


registered_models = {
    "base": ModelBase,
    "esd": DecompNet4ESD,
}



