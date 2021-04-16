import torch

from UNET.utils.loss import SoftDiceLoss, SoftIoULoss, BinaryLovaszLoss


class CustomLoss:
    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def __call__(self, predict, target):
        lovasz_loss = BinaryLovaszLoss()
        miou_loss = SoftIoULoss(logit=True, class_weights=None, log=False)
        dice_loss = SoftDiceLoss(logit=True, class_weights=None, log=False)
        bce_weight = self.class_weight[0] / self.class_weight[1] if self.class_weight is not None else None
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=bce_weight)
        # return 0.8*bce_loss(input,target) + lovasz_loss(input.squeeze(1),target.squeeze(1)) + dice_loss(input,target)
        return 0.8 * bce_loss(predict, target) + miou_loss(predict, target) + dice_loss(predict, target)
