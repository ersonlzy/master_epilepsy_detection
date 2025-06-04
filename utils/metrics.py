import torch
import torch.nn as nn
from torchmetrics import functional as f

class Metrics(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    @torch.no_grad()
    def forward(self, outputs):
        pred_logits = outputs['outputs']
        targets = outputs['targets']
        pred_probas = torch.softmax(pred_logits, -1)
        preds = torch.argmax(pred_probas, -1)
        trues = targets.long()
        if self.args.metric_type == "binary":
            pred_probas = pred_probas[:, 1]
        # print(preds.unique(), trues.unique(), self.args.num_classes)
        confmat = f.confusion_matrix(preds, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        prts = self.getPRTS(pred_probas, trues)
        map = torch.mean(f.average_precision(pred_probas, trues, task = self.args.metric_type, num_classes=self.args.num_classes))
        recall = f.recall(preds, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        specificity = f.specificity(preds, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        precision = f.precision(preds, trues, task = self.args.metric_type, num_classes=self.args.num_classes)
        accuracy = f.accuracy(preds, trues,  task = self.args.metric_type, num_classes=self.args.num_classes)    
        loss = torch.tensor(outputs['loss']).mean()
        return {"confmat": confmat,
                "map" : map,
                "prts" : prts,  
                "recall": recall,
                "precision": precision,
                "specificity": specificity,
                "accuracy": accuracy,
                "loss": loss,
                }
    
    def getPRTS(self, pred_probas, trues):
        precision_list = []
        recall_list = []
        th_list = []
        specificity_list = []
        for th in torch.range(0, 1.001, 0.01).float():
            th = th.item()
            precision = f.precision(pred_probas, trues, task = self.args.metric_type, num_classes=self.args.num_classes, threshold=th)
            recall = f.recall(pred_probas, trues, task = self.args.metric_type, num_classes=self.args.num_classes, threshold=th)
            specificity = f.specificity(pred_probas, trues, task = self.args.metric_type, num_classes=self.args.num_classes, threshold=th)
            precision_list.append(precision)
            recall_list.append(recall)
            th_list.append(th)
            specificity_list.append(specificity)
        return torch.tensor(precision_list), torch.tensor(recall_list), torch.tensor(th_list), torch.tensor(specificity_list)