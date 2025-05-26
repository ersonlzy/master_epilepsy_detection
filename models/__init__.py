import torch
import torch.nn as nn
from models.loss import *


registered_losses = {
    "ce": CrossEntropyLoss,
    "lr": FocalLoss,
    "lsce": LabelSmoothingCrossEntropyLoss,
}

class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.criterion = registered_losses[self.args.loss](args)

    def forward(self, x):
        raise NotImplementedError

    def kernel(self, samples, targets):
        outputs = self.forward(samples)
        loss = self.getLoss(outputs, targets)
        
        return {
            "loss": loss,
            "outputs": outputs,
        }

    def getLoss(self, outputs, targets):
        return self.criterion(outputs, targets)
    
