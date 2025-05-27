import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super(CrossEntropyLoss, self).__init__()
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)
    


# class FocalLoss(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.gamma = args.gamma  # 调节易分类样本的损失权重
#         self.alpha = args.alpha  # 权重参数，通常用来平衡类别不平衡
            
#     def forward(self, logits, targets):
#         # 使用标签平滑后计算交叉熵损失
#         ce_loss_smoothing = label_smooth(logits, targets, self.args.epsilon, logits.shape[-1])
#         # 计算概率 pt
#         pt = torch.exp(- ce_loss_smoothing)
#         # 计算 focal loss
#         focal_loss = (self.alpha * (1 - pt)**self.gamma * ce_loss_smoothing)
#         return focal_loss.mean()
    

class FocalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gamma = args.gamma
        self.alpha = args.alpha
        if isinstance(args.alpha,(float,int)): 
            self.alpha = torch.tensor([args.alpha,1-args.alpha], device=torch.device(self.args.device))
        if isinstance(args.alpha,list): 
            self.alpha = torch.tensor(args.alpha, device=torch.device(self.args.device))
        self.size_average = True



    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt) ** self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()


def label_smooth(logits, targets, epsilon, num_classes):
    # 构造平滑标签，除了正确类别为 1-beta，其他类别为 beta/(num_classes-1)
    t = torch.full_like(logits, epsilon/(num_classes-1), device=logits.device)
    t[targets.unsqueeze(-1)] = 1 - epsilon
    # 计算平滑后的交叉熵，返回每个样本的损失
    return F.cross_entropy(logits.softmax(-1), t, reduction='none')


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.epsilon = args.epsilon  # 平滑系数

    def forward(self, logits, targets):
        class_num = logits.shape[-1]
        # 计算 log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        # 将目标转换为 one-hot 向量
        targets_one_hot = F.one_hot(targets, num_classes=class_num).float()
        # 应用标签平滑
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / class_num
        # 计算加权的负对数似然损失
        loss = (-targets_smooth * log_probs).sum(dim=-1).mean()
        return loss

