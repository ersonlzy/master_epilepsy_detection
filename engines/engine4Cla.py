from engines import EngineBase
from alive_progress import alive_bar
from itertools import cycle
import torch
import math
from utils import *
import threading

class Engine4Cla(EngineBase):
    project = "classification"
    def __init__(self, args):
        super().__init__()


    def run(self):
        total_step = len(self.train_dataloader) * self.args.num_epochs / self.args.batch_size
        with alive_bar(total_step, title="training: ", ) as bar:
            train_outputs = None
            train_targets = None
            for i, samples, targets in enumerate(cycle(self.train_dataloader)):
                self.model.train()
                self.current_args.step = i + 1
                outputs = self.model.forward(samples, targets)
                self.optimizer.zero_grad()
                outputs["loss"].backward()
                if self.args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()
                
                train_outputs = self.concat(train_outputs, outputs)
                train_targets = self.concat(train_targets, targets)
                if self.current_args.step % self.args.log_step == 0:
                    self.log(train_outputs, train_targets, "train")
                    self.valid()
                    train_outputs = None
                    train_targets = None
                bar()
                if self.current_args.epoch == total_step:
                    break
                


    @torch.no_grad()
    def valid(self):
        self.model.eval()
        with alive_bar(math.ceil(len(self.valid_dataloader) / self.args.batch_size), title="validation: ", ) as bar:
            train_outputs = None
            train_targets = None
            for samples, targets in self.valid_dataloader:
                outputs = self.model.kernel(samples, targets)
                train_outputs = self.concat(train_outputs, outputs)
                train_targets = self.concat(train_targets, targets)
                bar()
            self.log(train_outputs, train_targets, "valid")


    @torch.no_grad()
    def concat(self, old, new):
        if old is None:
            old = new
        else:
            for key, value in new.items():
                old[key] = torch.concat(old[key], value, 0)
        return old



    def log(self, outputs, targets, tag):
        th = threading.Thread(target=_log, args=[self.swanlab, self.metrics, outputs, targets, tag, self.dataset.classes_list], daemon=True)
        th.start()
        self.swanlab.log()
        self.swanlab.Image(caption="random image")

def _log(logger, metric_holder, outputs, targets, tag, class_list):
    metrics = metric_holder(outputs, targets)
    logger.log({f"{tag}_loss": metrics["loss"], 
                f"{tag}_mAP": metrics["map"],
                f"{tag}_accuracy": metrics['accuracy'],
                f"{tag}_recall": metrics["recall"],
                f"{tag}_precision": metrics["precision"]}, print_to_console=True)
    confmat = metrics["confmat"]
    logger.log(logger.Image(confmat_plot(confmat, class_list)),  caption="Confusion Matrix")
    precision, recall, confidence = metrics["prt"]
    logger.log(logger.Image(precision_recall_plot(precision, recall, class_list)),  caption="Precision-Recall Curve")
    logger.log(logger.Image(precision_confidence_plot(precision, confidence, class_list)),  caption="Precision-Confidence Curve")
    logger.log(logger.Image(recall_confidence_plot(recall, confidence, class_list)),  caption="Recall-Confidence Curve")
    logger.log(logger.Image(f1score_confidence_plot(precision, recall, confidence, class_list)),  caption="F1score-Confidence Curve")





                    


        