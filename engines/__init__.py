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
import yaml
from argparse import Namespace
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
        self.device = torch.device(args.device)
        self.model = registered_models[args.model](args)
        print(self.args.__dict__)
        self.optimizer = registered_optimizers[self.args.optim](self.model.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        if self.args.resume:
            self.exp_path = os.path.join(*self.args.checkpoint.split("/")[:2])
            self.resumeExp()
        else:
            self.experiment_name = args.experiment if args.experiment else f"{self.project}_{args.model}"
            self.experiment_name = f"{self.experiment_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            self.args.experiment = self.experiment_name
            self.current_args.experiment = self.experiment_name
            self.createExp()

        self.swanlab = swanlab.init(
            project=self.project,
            experiment_name=self.args.experiment,
            config=self.args.__dict__,

        )

        self.dataset = registered_datasets[self.args.dataset](args)
        total_step = len(self.dataset) // self.args.batch_size * self.args.num_epochs
        self.lr_scheduler = registered_schedulers[self.args.scheduler](self.optimizer, T_max=total_step, eta_min=self.args.lr0, last_epoch=self.current_args.step - 1)
        self.metrics = Metrics(self.args)


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
        self.metrics = self.metrics.to(self.device)


    def saveExp(self):
        path = os.path.join(self.exp_path, "checkpoints")
        if not os.path.exists(path):
            os.mkdir(path)
        checkpoint_path = os.path.join(path, f"checkpoint-{self.current_args.step}")
        checkpoints = os.listdir(path)
        checkpoints = sorted(checkpoints, key=lambda x: os.path.getctime(os.path.join(path, x)))
        if len(checkpoints) >= self.args.num_saves:
            oldest = checkpoints[0]
            for file in os.listdir(os.path.join(path, oldest)):
                os.remove(os.path.join(path, oldest, file))
            os.rmdir(os.path.join(path, oldest))
        os.mkdir(checkpoint_path)

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        current_args = self.current_args.__dict__
        args = self.args.__dict__
        torch.save(model_state_dict, os.path.join(checkpoint_path, "model.pth"))
        torch.save(optim_state_dict, os.path.join(checkpoint_path, "optim.pth"))
        with open(os.path.join(checkpoint_path, "current_args.yaml"), "w") as f:
            yaml.dump(current_args, f)
        with open(os.path.join(checkpoint_path, "args.yaml"), "w") as f:
            yaml.dump(args, f)
        

        

    def createExp(self):
        if not os.path.exists("./runs"):
            os.mkdir("./runs")
        i = 0
        for folder in os.listdir("./runs"):
            if self.args.experiment in folder:
                i += 1
                continue
        self.exp_path = f"./runs/{self.args.experiment}"
        os.mkdir(self.exp_path)

    def resumeExp(self):
        if self.args.checkpoint:
            if not os.path.exists(self.args.checkpoint):
                raise FileNotFoundError(f"Checkpoint {self.args.checkpoint} not found")
            model_state_dict = torch.load(os.path.join(self.args.checkpoint, "model.pth"), map_location="cpu")
            optim_state_dict = torch.load(os.path.join(self.args.checkpoint, "optim.pth"), map_location="cpu")
            self.model.load_state_dict(model_state_dict)
            self.model = self.model.to(self.device)  # Move model to device after loading state dict
            self.optimizer.load_state_dict(optim_state_dict)
            # Move optimizer state to the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            args = yaml.safe_load(open(os.path.join(self.args.checkpoint, "args.yaml"), "r"))
            current_args = yaml.safe_load(open(os.path.join(self.args.checkpoint, "current_args.yaml"), "r"))
            for k in ["lr", "lr0", "weight_decay"]:
                if k in args:
                    args[k] = float(args[k])
                if k in current_args:
                    current_args[k] = float(current_args[k])
            print(f"Resuming experiment from checkpoint: {self.args.checkpoint}")
            self.args = Namespace(**args)
            self.current_args = Namespace(**current_args)
        else:
            raise ValueError("Checkpoint not specified. Please provide a valid checkpoint to resume from.")

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



