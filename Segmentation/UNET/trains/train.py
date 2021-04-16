import torch

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.engine import _prepare_batch

from tqdm import tqdm

from UNET.trains.utils import get_train_validation_data_loaders
from UNET import UNet, MultiClassSoftDiceMetric, SoftIOU, HardDice, HardIOU, MultiClassBCESoftDiceLoss

import os
import numpy as np

from PIL import Image
import segmentation_models_pytorch as smp


def load_trained_model(model, optimizer, model_path, optimizer_path):
    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
    print('Load model from: ', model_path)
    print('Load optimizer from: ', optimizer_path)


def save_model(model, optimizer, model_path, optimizer_path, postfix='_'):
    torch.save(model.state_dict(), model_path + postfix)
    torch.save(optimizer.state_dict(), optimizer_path + postfix)


def log_image(image, prefix, epoch, step):
    img = Image.fromarray(image)
    image_name = "%s_%s_%s.png" % (epoch, step, prefix)
    img.save(image_name)

    os.remove(image_name)


def run_test_model(model, evaluate_loader, epoch, device, step=10, log_to_mlflow=False):
    model.eval()
    count_step = 0

    for idx, batch in enumerate(evaluate_loader):
        if count_step > step:
            break

        x, y = _prepare_batch(batch, device)

        predict = model(x)
        predict = torch.sigmoid(predict) > 0.2

        for i in range(len(x)):
            gt = evaluate_loader.dataset.mask_to_grayscale(y[i])
            img = evaluate_loader.dataset.mask_to_grayscale(predict[i])


        count_step += len(x)

    model.train()


def load_classes_weights(dataset: str, cuda: bool, train_loader, nclass):
    classes_weights_path = os.path.join('/home/user/Unet/bears/datasets', dataset + '_classes_weights.npy')
    if os.path.isfile(classes_weights_path):
        weight = np.load(classes_weights_path)
    else:
        weight = calculate_weigths_labels(dataset, train_loader, nclass)
    weights = torch.from_numpy(weight.astype(np.float32))
    if cuda:
        weights = weights.cuda()
    return weights


def run_train(dataset_path, batch_size, n_processes, model_path, optimizer_path, load_pre_model=False,
              device='cpu', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0004, epochs=10,
              log_interval=20, save_interval=2, log_to_mlflow=False):

    train_loader, evaluate_loader = get_train_validation_data_loaders(path=dataset_path, batch_size=batch_size,
                                                                      n_processes=n_processes)
    model = smp.Unet(encoder_name='resnet50', classes=1)

    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available')

        model = model.to(device)
        print('CUDA is used')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    if load_pre_model:
        load_trained_model(model, optimizer, model_path, optimizer_path)

    # return
    trainer = create_supervised_trainer(model, optimizer, MultiClassBCESoftDiceLoss(0.7), device=device)

    nclass = 1

    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'valid_loss': Loss(MultiClassBCESoftDiceLoss(0.7)),
                                                # 'custom': Loss(CustomLoss(class_weight)),
                                                'soft_iou': SoftIOU(),
                                                'hard_iou': HardIOU(nclass + 1),
                                                'soft_dice': MultiClassSoftDiceMetric(),
                                                'hard_dice': HardDice(nclass + 1),

                                            },
                                            device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = None

    @trainer.on(Events.EPOCH_STARTED)
    def create_pbar(engine):
        model.train()
        nonlocal pbar
        pbar = tqdm(
            initial=0, leave=False, total=len(train_loader),
            desc=desc.format(0)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.close()
        evaluator.run(evaluate_loader)
        metrics = evaluator.state.metrics

        print("Training Results - Epoch: {}  Dice: {:.2f} Custom loss: {:.2f}"
              .format(engine.state.epoch, metrics['soft_dice'], metrics['valid_loss']))

        # print("Training Results - Epoch: {}  Dice: {:.2f} Avg loss: {:.2f}"
        #     .format(engine.state.epoch, avg_dice, avg_nll))

        if engine.state.epoch % save_interval == 0:
            save_model(model, optimizer, model_path, optimizer_path, '_' + str(engine.state.epoch))
            run_test_model(model, evaluate_loader, engine.state.epoch, device, log_to_mlflow=log_to_mlflow)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        pbar.desc = desc.format(engine.state.output)
        pbar.update()

    model.train()

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    run_train(dataset_path='/home/user/bears/datasets/data', batch_size=8, n_processes=8,
              model_path='/home/user/Unet/checkouts/model',
              optimizer_path='/home/user/Unet/checkouts/opt', device='cuda', epochs=100,
              load_pre_model=False, log_to_mlflow=False)
