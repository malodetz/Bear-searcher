import torch
import torch.nn as nn
import numpy as np
try:
    from itertools import ifilterfalse
except ImportError:     # py3k
    from itertools import filterfalse as ifilterfalse
from torch.autograd import Variable
import torch.nn.functional as F

SMOOTH = 1e-5


class SoftDiceLoss(nn.Module):

    def __init__(self, logit=True, class_weights=None, log=False):
        super(SoftDiceLoss, self).__init__()

        self.logit = logit
        self.class_weights = class_weights
        self.log = log

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Evaluate dice loss
        :param input: tensor of shape (B, C, W, H)
        :param target: tensor of shape (B, C, W, H)
        :return: tensor of shape (1,)
        """

        if self.logit:
            input = torch.sigmoid(input)

        numerator = 2. * torch.sum(input * target, dim=(2, 3))
        denominator = torch.sum(input * input, dim=(2, 3)) + torch.sum(target * target, dim=(2, 3))

        # calculating dice over (W,H) -> (B,C)
        dice = (numerator + SMOOTH) / (denominator + SMOOTH)

        # calculating class mean dice
        if self.class_weights:
            class_weights = torch.from_numpy(self.class_weights).repeat(dice.shape[0])
            dice *= class_weights
            dice = torch.sum(dice, dim=1)
        else:
            dice = torch.mean(dice, dim=1)

        # calculating batch mean dice
        dice = torch.mean(dice)

        # TODO: remove
        tmp = dice.cpu().detach().numpy()
        assert 0 <= tmp <= 1, "dice is out of bounds [1e-5, 1]: %f " % tmp

        # calculating dice loss
        if self.log:
            return -torch.log(dice)

        e = dice.new_ones([1])
        return e - dice


class SoftIoULoss(nn.Module):

    def __init__(self, logit=True, class_weights=None, log=False):
        super(SoftIoULoss, self).__init__()

        self.logit = logit
        self.class_weights = class_weights
        self.log = log

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Evaluate dice loss
        :param input: tensor of shape (B, C, W, H)
        :param target: tensor of shape (B, C, W, H)
        :return: tensor of shape (1,)
        """

        if self.logit:
            input = torch.sigmoid(input)
        # if target.shape[1] == 1:
        #     tmp = target.repeat(1, 2, 1, 1)
        #     tmpi = input.repeat(1, 2, 1, 1)
        #     tmp[:, 1, :, :] = 1 - target[:, 0, :, :]
        #     tmpi[:, 1, :, :] = 1 - input[:, 0, :, :]
        #     target = tmp
        #     input = tmpi

        intersection = torch.sum(input * target, dim=(2, 3))
        union = torch.sum(input * input, dim=(2, 3)) + torch.sum(target * target, dim=(2, 3)) - intersection

        # calculating mIoU index over (W,H) -> (B,C)
        mIoU = (intersection + SMOOTH) / (union + SMOOTH)

        # calculating class mean mIou index
        if self.class_weights is not None:
            mIoU *= self.class_weights
            mIoU = torch.mean(mIoU, dim=1)
        else:
            mIoU = torch.mean(mIoU, dim=1)

        # calculating batch mean mIoU index
        mIoU = torch.mean(mIoU)

        # TODO: remove
        tmp = mIoU.cpu().detach().numpy()
        assert 0 <= tmp <= 1, 'out of bounds [1e-5,1] %f' % tmp

        # calculating mIoU loss
        if self.log:
            mIoU = -torch.log(mIoU)
            return mIoU

        e = mIoU.new_ones([1])
        return e - mIoU


class SoftTverskyLoss(nn.Module):

    def __init__(self, alpha: float, beta: float, logit=True, class_weights=None, log=False):
        super(SoftTverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.logit = logit
        self.class_weights = class_weights
        self.log = log

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Evaluate dice loss
        :param input: tensor of shape (B, C, W, H)
        :param target: tensor of shape (B, C, W, H)
        :return: tensor of shape (1,)
        """

        if self.logit:
            input = torch.sigmoid(input)

        intersection = torch.sum(input * target, dim=(2, 3))
        denominator = intersection + self.alpha * torch.abs(input - target) + self.beta * torch.abs(target - input)

        # calculating tversky index over (W,H) -> (B,C)
        tversky = (intersection + SMOOTH) / (denominator + SMOOTH)

        # calculating class mean tversky index
        if self.class_weights:
            class_weights = torch.from_numpy(self.class_weights).repeat(tversky.shape[0])
            tversky *= class_weights
            tversky = torch.sum(tversky, dim=1)
        else:
            tversky = torch.mean(tversky, dim=1)

        # calculating batch mean tversky index
        tversky = torch.mean(tversky)

        # TODO: remove
        tmp = tversky.cpu().detach().numpy()
        assert 1e-5 <= tmp <= 1, "tversky index out of bounds [1e-5, 1]: %f" % tmp

        # calculating tversky loss
        if self.log:
            return -torch.log(tversky)

        e = tversky.new_ones([1])
        return e - tversky


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logit=True, class_weights=None, log=False):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.logit = logit
        self.class_weights = class_weights
        self.log = log

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.logit:
            input = torch.sigmoid(input)
        n, c, h, w = input.size()
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(input, target)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        loss /= n

        return loss


class BinaryLovaszLoss(nn.Module):

    def __init__(self, per_image=True, ignore=None):
        super(BinaryLovaszLoss, self).__init__()

        self.per_image = per_image
        self.ignore = ignore

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            loss = self.mean(self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                        for log, lab in zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.lovasz_hinge(logits=input, labels=target, per_image=self.per_image, ignore=self.ignore)


if __name__ == "__main__":
    loss = SoftIoULoss(logit=True, log=False)
    a = torch.ones(3, 2, 7, 7).cuda()
    b = torch.rand(3, 2, 7, 7).cuda()
    print(loss(a, b))
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




