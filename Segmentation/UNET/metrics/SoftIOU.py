from ignite.metrics import Metric
from UNET.utils.loss import SoftIoULoss


class SoftIOU(Metric):
    def __init__(self):
        super(SoftIOU, self).__init__()
        self.general_loss = 0

    def reset(self):
        self.general_loss = 0

    def update(self, output):
        predict, target = output
        iou = SoftIoULoss(logit=True, log=False)
        self.general_loss = 1 - iou(predict, target).cpu().detach().numpy()[0]

    def compute(self):
        return self.general_loss
