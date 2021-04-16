from UNET.metrics.BCESoftDiceLoss import BCESoftDiceLoss


class MultiClassBCESoftDiceLoss:
    def __init__(self, dice_weight=0):
        self.bce_soft_dice = BCESoftDiceLoss(dice_weight)

    def __call__(self, predict, target):
        classes = target.shape[1]
        loss = predict.new_zeros(1)

        for i in range(classes):
            loss += self.bce_soft_dice(predict[:, i].unsqueeze(1), target[:, i].unsqueeze(1))

        return loss[0] / float(classes)
