import torch
import segmentation_models_pytorch as smp
import albumentations as A
import cv2
from UNET.datasets.UNETDataset import tensor_from_rgb_image
from UNET.datasets.utils import get_color_map
import numpy as np
import os


def find_files(folder, extension):
    if folder is None:
        raise Exception('Finding of files failed')

    try:
        tree = os.walk(folder)
        files = []
        for _ in tree:
            files = files.__add__([{'path': os.path.join(_[0], f), 'name': f}
                                   for f in filter(lambda x: x[-len(extension[0]):].lower() in extension, _[2])])
    except WindowsError:
        raise Exception('Finding of files failed')
    return files


def correct_image_shape(image):
    if len(image.shape) == 2:
        return np.expand_dims(image, -1)
    return image


def correct_image_shape2(image, channels):
    if channels == 1 and len(image.shape) > 2:
        return image.reshape(image.shape[:2 - len(image.shape)])
    return image


def get_resized_image_in_rectangle3(image, sizes):
    if image is None:
        return None
    img = correct_image_shape(image)
    height, width, channels = img.shape
    rect = (0, 0, sizes[0], sizes[1])
    h = rect[3]
    w = int(h * width / height)
    if w < rect[2]:
        w = rect[2]
        h = int(height / width * w)
    new_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    s = np.full((rect[3], rect[2], channels), np.uint8(255))
    sy = int((rect[3] - h) / 2)
    sx = int((rect[2] - w) / 2)

    if sy < 0:
        sy2 = h - rect[3] + sy
        s[0:rect[3], 0:rect[2]] = correct_image_shape(new_img[-sy:h-sy2, 0:w])
    elif sx <= 0:
        sx2 = w - rect[2] + sx
        s[0:rect[3], 0:rect[2]] = correct_image_shape(new_img[0:h, -sx:w-sx2])
    else:
        s[sy:sy + h, sx:sx + w] = correct_image_shape(new_img)

    return correct_image_shape2(s, channels)


def mask_to_grayscale(masks) -> np.ndarray:
    masks = masks.cpu().numpy()

    colors_by_index = list(get_color_map().values())
    img = np.zeros(masks.shape[1:], dtype=np.uint8)

    for i in range(len(masks)):
        img[masks[i] == 1] = colors_by_index[i]

    return img


def get_test_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
    ], p=1)


def run_test(model_path, test_folder_path):
    model = smp.Unet(encoder_name='resnet50', classes=1)
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    images = find_files(test_folder_path, '.png')
    for image_file in images:
        # if image['name'] != '0029.png':
        #     continue
        img = cv2.imread(image_file['path'], 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = get_resized_image_in_rectangle3(img, [640, 512])

        transform = get_test_augmentation()
        image = transform(image=img)['image']
        image = tensor_from_rgb_image(image)
        # image = image.view((1, 3, image.shape[1], image.shape[2]))
        image = image.view((1, 3, image.shape[1], image.shape[2])).cuda()
        with torch.no_grad():
            predict = model(image)
        predict = torch.sigmoid(predict) > 0.2
        res = mask_to_grayscale(predict[0])

        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

        img = cv2.addWeighted(res, 0.3, img, 0.7, 0.0)

        cv2.imshow('img', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    run_test('/home/user/Unet/checkouts/model_64', '/home/user/bears/datasets/data/val/image')
