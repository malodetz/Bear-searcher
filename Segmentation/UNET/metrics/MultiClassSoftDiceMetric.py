from ignite.metrics import Metric

from UNET.metrics.BCESoftDiceLoss import BCESoftDiceLoss


class MultiClassSoftDiceMetric(Metric):
    def __init__(self):
        super(MultiClassSoftDiceMetric, self).__init__()
        self.general_loss = 0

    def reset(self):
        self.general_loss = 0

    def update(self, output):
        predict, target = output

        classes = target.shape[1]
        loss = predict.new_zeros(1)

        for i in range(classes):
            loss += BCESoftDiceLoss.soft_dice(predict[:, i].unsqueeze(1), target[:, i].unsqueeze(1))

        self.general_loss = loss[0] / float(classes)

    def compute(self):
        return self.general_loss
