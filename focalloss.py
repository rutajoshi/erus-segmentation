import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def compute_class_weight(labels, num_classes):
    """Make class weights for cross-entropy loss (useful for class imbalance,)
    in accordance with sklearn.utils.class_weight.compute_class_weight"""
    weights = len(labels) / (num_classes * torch.bincount(labels))
    weights = weights / sum(weights)
    return weights

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim()>2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)
            inputs = inputs.transpose(1,2)
            inputs = inputs.contiguous().view(-1,inputs.size(2))
        target = target.view(-1,1)

        logpt = F.log_softmax(inputs)
        logpt = logpt.gather(1,target.clone())
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0,target.clone().data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
