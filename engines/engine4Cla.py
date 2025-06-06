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
    project = "master_epilepsy_detection"
    def __init__(self, args):
        super().__init__(args)
    

    def run(self):
        self.expReady()
        print(self.dataset.splitReport())
        print(len(self.train_dataloader), len(self.valid_dataloader))
        total_step = (len(self.train_dataloader) * self.args.num_epochs)
        epoch_per_step = self.args.num_epochs / total_step
        with alive_bar(total_step, title="experiment is progressing: ", ) as bar:
            bar(self.current_args.step + 1)
            try:
                train_outputs = None
                for samples, targets in cycle(self.train_dataloader):
                    self.model.train()
                    self.current_args.step += 1
                    self.current_args.epoch = round(self.current_args.step * epoch_per_step, 2)
                    outputs = self.model.kernel(samples, targets)
                    self.optimizer.zero_grad()
                    outputs['loss'].backward()
                    
                    if self.args.max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                        self.optimizer.step()
                    
                    train_outputs = self.concat(train_outputs, outputs)
                    if self.current_args.step % self.args.log_step == 0:
                        self.log(train_outputs, "train")
                        self.saveExp()
                        self.valid()
                        train_outputs = None
                    self.lr_scheduler.step()
                    bar()
                    if self.current_args.step == total_step:
                        break
            except KeyboardInterrupt:
                print("KeyboardInterrupt: manually stopping training")
                self.saveExp()
        swanlab.finish(self.swanlab)
                


    @torch.no_grad()
    def valid(self):
        self.model.eval()
        train_outputs = None
        for samples, targets in self.valid_dataloader:
            outputs = self.model.kernel(samples, targets)
            train_outputs = self.concat(train_outputs, outputs)
        self.log(train_outputs, "valid")


    @torch.no_grad()
    def concat(self, old, new):
        if old is None:
            old = new
            old['loss'] = [new['loss']]
        else:
            for key, value in new.items():
                if key == "loss":
                    old[key].append(value.detach())
                else:
                    old[key] = torch.concat([old[key], value.detach()], 0)
        return old



    def log(self, outputs, tag):
        _log(swanlab, self.metrics, outputs, tag, self.dataset.classes_list, self.current_args)


@torch.no_grad()
def _log(logger, metric_holder, outputs, tag, class_list, args):
    metrics = metric_holder(outputs)
    confmat = metrics["confmat"]
    logger.log({f"{tag}/Confusion Matrix": logger.Image(confmat_plot(confmat, class_list), caption="Confusion Matrix")})
    precision, recall, confidence, specificity = metrics["prts"]
    # if args.metric_type == "binary":
    precision = [precision]
    recall = [recall]
    confidence = [confidence]
    specificity = [specificity]
    class_list = [class_list[-1]]
    logger.log({f"{tag}/loss": round(metrics["loss"].item(), 4), 
                f"{tag}/mAP": round(metrics["map"].item(), 4),
                f"{tag}/auc": round(metrics["auc"].item(), 4),
                f"{tag}/accuracy": round(metrics['accuracy'].item(), 4),
                f"{tag}/recall": round(metrics["recall"].item(), 4),
                f"{tag}/specificity": round(metrics["specificity"].item(), 4),
                f"{tag}/precision": round(metrics["precision"].item(), 4)}, print_to_console=True, step=args.step)
    # logger.log({f"{tag}/Precision-Recall Curve": logger.Image(precision_recall_plot(precision, recall, class_list), caption="Precision-Recall Curve")})
    # logger.log({f"{tag}/Precision-Confidence Curve": logger.Image(precision_confidence_plot(precision, confidence, class_list), caption="Precision-Confidence Curve")})
    # logger.log({f"{tag}/Recall-Confidence Curve": logger.Image(recall_confidence_plot(recall, confidence, class_list), caption="Recall-Confidence Curve")})
    # logger.log({f"{tag}/Specificity-Confidence Curve": logger.Image(specificity_confidence_plot(specificity, confidence, class_list), caption="Specificity-Confidence Curve")})
    # logger.log({f"{tag}/F1score-Confidence Curve": logger.Image(f1score_confidence_plot(precision, recall, confidence, class_list), caption="F1score-Confidence Curve")})
    # loss = torch.nn.functional.cross_entropy(outputs["outputs"], outputs['targets'], reduction="none")
    # r = torch.min(loss, 0)
    # _, min_index = r
    # decomp_out = outputs["decomp_out"][min_index]
    # diff_out = outputs["diff_out"][min_index]
    # mixer_out = outputs["mixer_out"][min_index]
    # feats = outputs["feats"]
    # logger.log({f"{tag}/decomp_out": logger.Image(decomp_out_plot(decomp_out), caption="decomp_out")})
    # logger.log({f"{tag}/diff_out": logger.Image(diff_out_plot(diff_out), caption="diff_out")})
    # logger.log({f"{tag}/mixer_out": logger.Image(mixer_out_plot(mixer_out), caption="mixer_out")})
    # logger.log({f"{tag}/feats": logger.Image(feats_plot(feats), caption="feats")})


    





                    


        