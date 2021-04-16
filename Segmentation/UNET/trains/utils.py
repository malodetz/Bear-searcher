import os

from albumentations import OneOf, GaussNoise, IAAAdditiveGaussianNoise, ShiftScaleRotate

from UNET import UNETDataset, BearsDataset

import albumentations as A

from torch.utils.data import DataLoader


def get_training_augmentation():
    return A.Compose([
        # A.RandomCrop(height=256, width=256, p=1),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        OneOf([
            # A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            # ShiftScaleRotate(p=0.5)
        ]),
        # OneOf([
        #     IAAAdditiveGaussianNoise(p=0.9),
        #     GaussNoise(p=0.6),
        # ], p=0.2)
    ], p=1)


def get_training_augmentation2():
    return A.Compose(
        [
            A.Normalize(mean=(0.5,), std=(0.5,)),
            A.HorizontalFlip(p=0.5),  # apply horizontal flip to 50% of images
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(),  # apply random contrast
                ],
                p=0.5
            ),
            A.OneOf(
                [
                    # apply one of transforms to 50% images
                    A.ElasticTransform(
                        alpha=120,
                        sigma=120 * 0.05,
                        alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(),
                    A.OpticalDistortion(
                        distort_limit=2,
                        shift_limit=0.5
                    ),
                ],
                p=0.3
            )
        ],
        p=1
    )


def get_test_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
    ], p=1)


def get_data_loader(path, batch_size, n_processes, shuffle=True):
    image_path = os.path.join(path, 'image')
    mask_path = os.path.join(path, 'mask')

    dataset = BacteriaDataset(image_folder=image_path, mask_folder=mask_path, transform=get_training_augmentation())

    return DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, num_workers=n_processes, shuffle=shuffle)


def get_train_validation_data_loaders(path, batch_size, n_processes):
    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'val')

    train_dl = get_data_loader(train_path, batch_size, n_processes, shuffle=True)
    test_dl = get_data_loader(valid_path, batch_size, n_processes, shuffle=False)

    return train_dl, test_dl
