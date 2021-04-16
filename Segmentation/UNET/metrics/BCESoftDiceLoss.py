import torch
import torch.nn as nn


class BCESoftDiceLoss:
    def __init__(self, dice_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    @staticmethod
    def soft_dice(predict, target):
        eps = 1e-15
        batch_size = target.size()[0]

        dice_target = (target == 1).float().view(batch_size, -1)
        dice_predict = torch.sigmoid(predict).view(batch_size, -1)

        inter = torch.sum(dice_predict * dice_target) / batch_size
        union = (torch.sum(dice_predict) + torch.sum(dice_target)) / batch_size + eps

        return (2 * inter.float() + eps) / union.float()

    def __call__(self, predict, target):
        loss = (1.0 - self.dice_weight) * self.nll_loss(predict, target)

        if self.dice_weight:
            loss -= self.dice_weight * torch.log(BCESoftDiceLoss.soft_dice(predict, target))

        return loss
