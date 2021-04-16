from ignite.metrics import Metric
import numpy as np


class HardIOU(Metric):
    def __init__(self, num_class):
        self.num_class = num_class
        super(HardIOU, self).__init__()
        self.confusion_matrix = np.zeros((num_class,) * 2)
        self.general_loss = 0

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.general_loss = 0

    def generate_matrix(self, pre_image, gt_image):
        confusion_matrix = np.zeros((self.num_class,) * 2)
        gt_image = gt_image.cpu().detach().numpy()
        pre_image = pre_image.cpu().detach().numpy()
        for i in range(gt_image.shape[0]):
            if self.num_class is not 2:
                gt_mask = np.argmax(gt_image[i], axis=0)
                pre_mask = np.argmax(pre_image[i], axis=0)
            else:
                gt_mask = gt_image[i]
                pre_mask = pre_image[i] > 0.8
            mask = (gt_mask >= 0) & (gt_mask < self.num_class)
            label = self.num_class * gt_mask[mask].astype('int') + pre_mask[mask]
            count = np.bincount(label, minlength=self.num_class ** 2)
            confusion_matrix += count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update(self, output):
        predict, target = output
        self.confusion_matrix = self.generate_matrix(predict, target)
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection

        MIoU = intersection / union
        MIoU = np.nanmean(MIoU)

        self.general_loss = MIoU

    def compute(self):
        return self.general_loss
