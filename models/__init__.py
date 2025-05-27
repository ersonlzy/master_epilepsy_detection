import torch
import torch.nn as nn
from models.loss import *


registered_losses = {
    "ce": CrossEntropyLoss,
    "fl": FocalLoss,
    "lsce": LabelSmoothingCrossEntropyLoss,
}

class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.criterion = registered_losses[self.args.loss](args)

    def forward(self, x):
        raise NotImplementedError

    def kernel(self, samples, targets):
        samples = samples.to(self.device)
        targets = targets.to(self.device)
        outputs = self.forward(samples)
        print(outputs, targets)
        loss = self.criterion(outputs, targets)
        
        return {
            "loss": loss,
            "outputs": outputs,
            'targets': targets,
        }
    
