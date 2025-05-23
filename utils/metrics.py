import torch
import torch.nn as nn
from torchmetrics import functional as f

class Metrics(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits = outputs['logits']
        pred_probas = torch.softmax(pred_logits, -1)
        preds = torch.argmax(pred_probas, -1)
        trues = targets['labels']
        confmat = f.confusion_matrix(preds, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        prt = f.precision_recall_curve(pred_probas, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        map = torch.mean(f.average_precision(pred_probas, trues, task = self.args.metric_type, num_classes=self.args.num_classes))
        recall = f.recall(preds, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        precision = f.precision(preds, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        accuracy = f.accuracy(preds, trues,  task = self.args.metric_type, num_classes=self.args.num_classes)    
        loss = nn.functional.cross_entropy(pred_logits, trues)
        return {"confmat": confmat,
                "map" : map,
                "prt" : prt,  
                "recall": recall,
                "precision": precision,
                "accuracy": accuracy,
                "loss": loss,
                }