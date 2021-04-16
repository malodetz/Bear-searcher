from ignite.metrics import Metric
import torch


class CE(Metric):
    def __init__(self, class_weights):
        super(CE, self).__init__()
        self.class_weights = class_weights
        self.general_loss = 0

    def reset(self):
        self.general_loss = 0

    def update(self, output):
        predict, target = output
        ce = torch.nn.BCEWithLogitsLoss(weight=self.class_weights)
        self.general_loss = ce(predict, target).cpu().detach().numpy()

    def compute(self):
        return self.general_loss
