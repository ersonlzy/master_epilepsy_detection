from engines import EngineBase
from alive_progress import alive_bar, alive_it
from itertools import cycle
import torch
import math
from utils import *
import threading
import swanlab
from tqdm import tqdm


class Engine4Cla(EngineBase):
    project = "classification"
    def __init__(self, args):
        super().__init__(args)
    

    def run(self):
        self.expReady()
        total_step = len(self.train_dataloader) // self.args.batch_size * self.args.num_epochs
        epoch_per_step = self.args.num_epochs / total_step
        # step_loss_list = []
        # with alive_bar(total_step, title="experiment is progressing: ", ) as bar:
        #     bar()
        train_outputs = None
        train_targets = torch.tensor([])
        for i, (samples, targets) in tqdm(enumerate(cycle(self.train_dataloader)), desc="Training", leave=False, total=total_step):
            self.model.train()
            self.current_args.step = i + 1
            self.current_args.epoch = round(self.current_args.step * epoch_per_step, 2)
            outputs = self.model.kernel(samples, targets)
            self.optimizer.zero_grad()
            outputs['loss'].backward()
            
            if self.args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()
            
            train_outputs = self.concat(train_outputs, outputs)
            train_targets = torch.concat([train_targets, targets])
            if self.current_args.step % self.args.log_step == 0:
                self.log(train_outputs, train_targets, "train")
                self.saveExp()
                self.valid()
                train_outputs = None
                train_targets = torch.tensor([])
            self.lr_scheduler.step()
            # bar()
            if self.current_args.step == total_step:
                break
        swanlab.finish(self.swanlab)
                


    @torch.no_grad()
    def valid(self):
        self.model.eval()
        train_outputs = None
        train_targets = torch.tensor([])
        for samples, targets in tqdm(self.valid_dataloader, desc="Validating", leave=False):
            outputs = self.model.kernel(samples, targets)
            train_outputs = self.concat(train_outputs, outputs)
            train_targets = torch.concat([train_targets, targets])
        self.log(train_outputs, train_targets, "valid")


    @torch.no_grad()
    def concat(self, old, new):
        if old is None:
            old = new
            old['loss'] = [new['loss']]
        else:
            for key, value in new.items():
                if key == "loss":
                    old[key].append(value)
                else:
                    old[key] = torch.concat([old[key], value], 0)
        return old



    def log(self, outputs, targets, tag):
        _log(swanlab, self.metrics, outputs, targets, tag, self.dataset.classes_list, self.current_args)


def _log(logger, metric_holder, outputs, targets, tag, class_list, args):
    metrics = metric_holder(outputs, targets)
    confmat = metrics["confmat"]
    logger.log({f"{tag}_Confusion Matrix": logger.Image(confmat_plot(confmat, class_list), caption="Confusion Matrix")})
    precision, recall, confidence, specificity = metrics["prts"]
    if args.metric_type == "binary":
        precision = [precision]
        recall = [recall]
        confidence = [confidence]
        specificity = [specificity]
        class_list = [class_list[-1]]
    logger.log({f"{tag}_loss": round(metrics["loss"].item(), 4), 
                f"{tag}_mAP": round(metrics["map"].item(), 4),
                f"{tag}_accuracy": round(metrics['accuracy'].item(), 4),
                f"{tag}_recall": round(metrics["recall"].item(), 4),
                f"{tag}_specificity": round(metrics["specificity"].item(), 4),
                f"{tag}_precision": round(metrics["precision"].item(), 4),
                "epoch": args.epoch}, print_to_console=True)
    logger.log({f"{tag}_Precision-Recall Curve": logger.Image(precision_recall_plot(precision, recall, class_list), caption="Precision-Recall Curve")})
    logger.log({f"{tag}_Precision-Confidence Curve": logger.Image(precision_confidence_plot(precision, confidence, class_list), caption="Precision-Confidence Curve")})
    logger.log({f"{tag}_Recall-Confidence Curve": logger.Image(recall_confidence_plot(recall, confidence, class_list), caption="Recall-Confidence Curve")})
    logger.log({f"{tag}_Specificity-Confidence Curve": logger.Image(specificity_confidence_plot(specificity, confidence, class_list), caption="Specificity-Confidence Curve")})
    logger.log({f"{tag}_F1score-Confidence Curve": logger.Image(f1score_confidence_plot(precision, recall, confidence, class_list), caption="F1score-Confidence Curve")})
    





                    


        