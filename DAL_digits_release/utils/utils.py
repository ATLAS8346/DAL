import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_rec(src, trg):
    return torch.sum((src - trg)**2) / (src.shape[0] * src.shape[1])


def _ent(out):
    return - torch.mean(torch.log(F.softmax(out + 1e-6, dim=-1)))


def _discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))


def _ring(feat, type='geman'):
    x = feat.pow(2).sum(dim=1).pow(0.5)
    radius = x.mean()
    radius = radius.expand_as(x)
    # print(radius)
    if type == 'geman':
        l2_loss = (x - radius).pow(2).sum(dim=0) / (x.shape[0] * 0.5)
        return l2_loss
    else:
        raise NotImplementedError("Only 'geman' is implemented")


class RingLoss(nn.Module):
    def __init__(self, type='auto', loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x_norm = x.pow(2).sum(dim=1).pow(0.5)
        R = self.radius.expand_as(x_norm)

        if self.radius.data[0] < 0:
            # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x_norm.mean().data)
        if self.type == 'l1':
            # Smooth L1 Loss
            loss1 = self.loss_weight * F.smooth_l1_loss(x_norm, R)
            loss2 = self.loss_weight * F.smooth_l1_loss(R, x_norm)
            ring_loss = loss1 + loss2
        elif self.type == 'auto':
            # Divide the L2 Loss by the feature's own norm
            diff = (x_norm - R) / (x_norm.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ring_loss = self.loss_weight * diff_sq
        else:
            # L2 Loss, if not specified
            diff_sq = torch.pow(torch.abs(x_norm - R), 2).mean()
            ring_loss = self.loss_weight * diff_sq
        return ring_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot
