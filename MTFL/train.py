import numpy as np
import torch
import torch.nn as nn
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss



def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_regularization(model, l1_alpha):
    l1_loss = 0.0
    for param in model.parameters():
        if param.dim() > 1:  # Only apply L1 regularization to parameters with more than one dimension (e.g., weights, not biases)
            l1_loss += torch.abs(param).sum()
    return l1_alpha * l1_loss


# def l2_regularization(model, l2_alpha):
#     l2_loss = []
#     for module in model.modules():
#         if type(module) is nn.Conv2d:
#             l2_loss.append((module.weight ** 2).sum() / 2.0)
#     return l2_alpha * sum(l2_loss)


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total




def train(nloader, aloader, model, batch_size, seg_num, optimizer, scheduler, viz, device):
    with torch.set_grad_enabled(True):
        model.train()

        ninput1, ninput2, ninput3, nlabel = next(nloader)
        ainput1, ainput2, ainput3, alabel = next(aloader)

        input1 = torch.cat((ninput1, ainput1), 0).to(device)
        input2 = torch.cat((ninput2, ainput2), 0).to(device)
        input3 = torch.cat((ninput1, ainput3), 0).to(device)
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input1, input2, input3)  # b*32  x 2048

        scores = scores.view(batch_size * seg_num * 2, -1)

        scores = scores.squeeze()
        abn_scores = scores[batch_size * seg_num:]
        nor_scores = scores[0: batch_size * seg_num]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        #l1_loss = l1_regularization(model, 0.0001)
        loss_RTFM = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)
        cost = loss_RTFM + loss_smooth + loss_sparse

        viz.plot_lines('loss', cost.item())
        viz.plot_lines('RTFM loss', loss_RTFM.item())
        viz.plot_lines('smooth loss', loss_smooth.item())
        viz.plot_lines('sparsity loss', loss_sparse.item())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        #scheduler.step()


