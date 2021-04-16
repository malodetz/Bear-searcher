import numpy as np
import torch
from UNET.utils.loss import SoftDiceLoss, SoftIoULoss


class Evaluator(object):
    def __init__(self, metrics, class_weights, num_class):

        self.metrics = metrics
        self.num_class = 2 if num_class else num_class
        self.class_weights = class_weights
        self.counter = 0
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.metrics_dict = {
            'soft_iou': self.soft_iou_index,
            'soft_dice': self.soft_dice_index,
            'ce': self.cross_entropy,
            'hard_iou': self.hard_iou,
            'hard_dice': self.hard_dice,
            'acc': self.pixel_accuracy,
            'mean_acc': self.mean_pixel_accuracy,
            'class_acc': self.classwise_pixel_accuracy,
            'fwiou': self.frequency_weighted_iou,
        }
        self.metrics_res = dict.fromkeys(self.metrics_dict.keys(), 0.0)
        if any([metric not in self.metrics_dict for metric in self.metrics]):
            raise NotImplementedError

    def add_batch(self, input, target):
        self.confusion_matrix = self.generate_matrix(input, target)
        for metric in self.metrics:
            self.metrics_res[metric] += self.metrics_dict[metric](input, target)

        self.counter += 1

    def reset(self):
        self.counter = 0
        self.metrics_res = dict.fromkeys(self.metrics_dict.keys(), 0.0)
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

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

    def eval_metrics(self, val_loss_info, show=True):
        metrics_results = []
        for key in self.metrics:
            self.metrics_res[key] /= self.counter
            metrics_results.append("%s:%f" % (key, self.metrics_res[key]))
        if show:
            print(val_loss_info, ': ', *metrics_results, sep=", ")
        return self.metrics_res

    def cross_entropy(self, input, target):
        ce = torch.nn.BCEWithLogitsLoss(weight=self.class_weights)
        return ce(input, target).cpu().detach().numpy()

    def soft_iou_index(self, input, target):
        iou = SoftIoULoss(logit=True, class_weights=self.class_weights, log=False)
        return 1 - iou(input, target).cpu().detach().numpy()[0]

    def soft_dice_index(self, input, target):
        dice = SoftDiceLoss(logit=True, class_weights=self.class_weights, log=False)
        return 1 - dice(input, target).cpu().detach().numpy()[0]

    def pixel_accuracy(self, input, target):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def mean_pixel_accuracy(self, input, target):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def classwise_pixel_accuracy(self, input, target):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return Acc

    def hard_iou(self, input, target):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection

        MIoU = intersection / union
        MIoU = np.nanmean(MIoU)
        return MIoU

    def hard_dice(self, input, target):
        intersection = 2 * np.diag(self.confusion_matrix)
        denominator = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
        dice = intersection / denominator
        dice = np.nanmean(dice)
        return dice

    def frequency_weighted_iou(self, input, target):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
