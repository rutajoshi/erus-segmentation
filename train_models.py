import argparse, sys, os
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST as Dataset
import torch.optim as optim

from unet.unet import UNet
from unet.logger2 import Logger
from unet.dataset import BrainSegmentationDataset

from mask_rcnn.mask_rcnn_models import maskrcnn_model

HAS_CUDA = torch.cuda.is_available()
device = torch.device("cpu" if not HAS_CUDA else "cuda:0")


CONFIG_CIFAR = { # config for cifar10
    "image_size": 34,
    "padding": 1,
    "units": [3, 32, 32,
              64, 64, 128, 128, 64, 64, 64, 64, 64],
    "train_data": torchvision.datasets.CIFAR10('../cifar', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Pad(1),
        torchvision.transforms.Resize(224),
    ])),
    "test_data": torchvision.datasets.CIFAR10('../cifar', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Pad(1),
        torchvision.transforms.Resize(224),
    ]))
}

CONFIG_MNIST = { # config for MNIST
    "image_size": 28,
    "padding": 0,
    "units": [1, 32, 32,
              64, 64, 128, 128, 64, 64, 64],
    "train_data": torchvision.datasets.MNIST('../mnist', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224), # for unet compatibility
    ])),
    "test_data": torchvision.datasets.MNIST('../mnist', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224), # for unet compatibility
    ]))
}

CONFIG_BRAINMRI = { # config for brain MRI dataset
    "image_size": 256,
    "padding": 0,
    "units": [3, 32, 64, 128, 256, 512, 256, 128, 64, 32, 1],
    "train_data": BrainSegmentationDataset('data/brain-mri', subset='train', transform=None),
    "test_data": BrainSegmentationDataset('data/brain-mri', subset='validation', transform=None)
}

def get_dataloader(config, batch_size=128):
    # Get train and test data
    train_data = config["train_data"]
    test_data = config["test_data"]

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def load_dataset(dataset_name, split, batch_size):
    train_loader, test_loader, config = None, None, None
    if (dataset_name == 'mnist'):
        config = CONFIG_MNIST
    elif (dataset_name == 'cifar'):
        config = CONFIG_CIFAR
    elif (dataset_name == 'brain'):
        config = CONFIG_BRAINMRI
    train_loader, test_loader = get_dataloader(config, batch_size)
    dataloaders = {"train": train_loader, "valid": test_loader}
    return dataloaders

def load_model(model_name, dataset_name):
    # Default
    if (model_name == 'unet'):
        if (dataset_name.lower() in ['mnist']):
            model = UNet(in_channels=1, out_channels=2)
        if (dataset_name in ['cifar', 'brain']):
            model = UNet(in_channels=3, out_channels=2)
    elif (model_name == 'gnn'):
        agg = GCNNAgg((4,4), agg_method='max')
        gnn = BaseGNN(5, None, CONFIG["units"], agg, torch.nn.functional.relu)
        seg_head = GNNSeg(gnn, 10)
        model = seg_head
    elif (model_name == 'mask_rcnn'):
        model = maskrcnn_model()
    return model

def load_loss(loss_name, gamma=2):
    loss = torch.nn.CrossEntropyLoss()
    if (loss_name == "FL"):
        loss = FocalLoss()
    return loss

def load_optimizer(params, optimizer_name, learning_rate):
    optimizer = optim.SGD(params, lr=learning_rate)
    if (optimizer_name == "Adam"):
        optimizer = optim.Adam(params, lr=learning_rate)
    return optimizer

# Utility function from unet/train.py
def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)

def train(model, dataloaders, loss_function, optimizer,
          logger, save_freq, save_path, num_epochs=100, has_mask=True):
    """
    By default: train forever
    """
    epoch = 0
    step = 0
    loss_train, loss_valid = [], []
    while epoch < num_epochs:
        print("Starting epoch " + str(epoch))
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            # Get correct dataloader for the phase
            loader = dataloaders[phase]

            for i, data in enumerate(loader):
                if phase == "train":
                    step += 1

                x, y_classes = data # x has shape [N, 1, 224, 224] --> y_true [N, 224, 224]
                # For either dataset, the binary image is the segmentation mask
                # Remove channel dimension, then binarize from float to long
                # because long() would round all decimals down to 0
                if not has_mask:
                    y_true = (x > 0.5).long()
                x, y_true = x.to(device), y_true.to(device).long()[:,0]
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # There should be one output channel for each segmentation group
                    y_pred = model(x)

                    loss = loss_function(y_pred, y_true)

                    if (step % 100 == 0 and phase != "valid"):
                        print("Loss at step " + str(step) + " = " + str(loss.detach().cpu().numpy()))

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )

                    if (phase == "train"):
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()


                if (phase == "train" and (step + 1) % 10 == 0):
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                if (epoch % save_freq == 0):
                    torch.save(model.state_dict(), os.path.join(save_path, "model_"+str(epoch)+".pt"))
                loss_valid = []

        epoch += 1
        if (epoch % 10 == 0):
            print("On epoch: " + str(epoch))


def train_maskrcnn(model, dataloaders, loss_function, optimizer, logger, save_freq, save_path, num_epochs=100):
    """
    By default: train forever
    """
    epoch = 0
    step = 0
    loss_train, loss_valid = [], []
    while epoch < num_epochs:
        print("Starting epoch " + str(epoch))
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            # Get correct dataloader for the phase
            loader = dataloaders[phase]

            for i, data in enumerate(loader):
                if phase == "train":
                    step += 1

                x, y_classes = data # x has shape [N, 1, 224, 224] --> y_true [N, 224, 224]
                # For either dataset, the binary image is the segmentation mask
                # Remove channel dimension, then binarize from float to long
                # because long() would round all decimals down to 0
                y_true = (x[:,0] > 0.5).long()
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # There should be one output channel for each segmentation group
                    targets = [{
                        "boxes": torch.Tensor([[0, 0, 28, 28]]),
                        "masks": y_true[i],
                        "labels": y_classes[i]
                    } for i in range(x.shape[0])]

                    loss = model(x, targets)

                    if (step % 100 == 0 and phase != "valid"):
                        print("Loss at step " + str(step) + " = " + str(loss.detach().cpu().numpy()))

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        # y_pred_np = y_pred.detach().cpu().numpy()
                        # validation_pred.extend(
                        #     [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        # )
                        # y_true_np = y_true.detach().cpu().numpy()
                        # validation_true.extend(
                        #     [y_true_np[s] for s in range(y_true_np.shape[0])]
                        # )

                    if (phase == "train"):
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if (phase == "train" and (step + 1) % 10 == 0):
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                if (epoch % save_freq == 0):
                    torch.save(model.state_dict(), os.path.join(save_path, "model_"+str(epoch)+".pt"))
                loss_valid = []

        epoch += 1
        if (epoch % 10 == 0):
            print("On epoch: " + str(epoch))


import matplotlib.pyplot as plt

def inference(model, dataloader, loss_function, logger, has_mask=True):
    # Go through each image and do inference, storing predictions somewhere
    loss_infer = []
    for i, data in enumerate(dataloader):
        x, y_true = data
        if not has_mask:
            y_true = (x > 0.5).long()
        x, y_true = x.to(device), y_true.to(device)[:,0].long()
        with torch.set_grad_enabled(False):
            y_pred = model(x)
            loss = loss_function(y_pred, y_true)
            loss_infer.append(loss)

            tag = "image/{}".format(i)
            logger.image_list_summary(
                tag,
                logger.log_images(x, y_true, y_pred),
                i, # step = i
            )
            #print(y_pred.shape)
            #plt.imshow((y_pred.detach().cpu().numpy()[0,1] * 256).astype(np.uint8))
            #plt.show()
        print(i)
    return loss_infer

# Train a model given the model name, dataset, loss function, and other parameters
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', help='Model type', type=str, default='mask_rcnn')
    parser.add_argument('--dataset', help='Dataset name', type=str, default='mnist')
    parser.add_argument('--loss', help='Loss function (CE/FL/DL)', type=str, default='CE')
    parser.add_argument('--gamma', help='gamma for focal loss', type=int, default=2)
    parser.add_argument('--optimizer', help='SGD/Adam', type=str, default='SGD')
    parser.add_argument('--learning_rate', help='Learning rate', type=int, default=0.01)
    parser.add_argument('--split', help='Train/valid/test', type=str, default='train')
    parser.add_argument('--inference', help='boolean inference', type=bool, default=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--device', help='cuda if available', type=str, default='cpu')
    parser.add_argument('--log_dir', help='directory for logs', type=str, default='./logs')
    parser.add_argument('--save_freq', help='how often to save model', type=int, default=10)
    parser.add_argument('--save_path', help='directory for model saving', type=str, default='./logs')
    parser.add_argument('--epochs', help='number of epochs', type=int, default=100)
    parser.add_argument('--saved_weights', help='path to saved model for inference', type=str, default='./logs/model_100.pt')
    args=parser.parse_args()

    # Pick device

    # Initialize logger
    logger = Logger(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load dataset
    dataloaders = load_dataset(args.dataset, args.split, args.batch_size)
    has_mask = (args.dataset not in ['mnist'])
    # Load model
    model = load_model(args.model, args.dataset)

    # Load loss function
    loss = load_loss(args.loss, args.gamma)
    # Load optimizer
    optimizer = load_optimizer(model.parameters(), args.optimizer, args.learning_rate)

    # Train or do inference
    if (args.inference):
        state_dict = torch.load(args.saved_weights, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        loss_infer = []
        if (args.split == "train"):
            loss_infer = inference(model, dataloaders["train"], loss, logger, has_mask)
        elif (args.split == "test"):
            loss_infer = inference(model, dataloaders["valid"], loss, logger, has_mask)
        print("Inference loss per image = " + str(loss_infer))
    else:
        model.to(device)
        if args.model == "mask_rcnn":
            train_maskrcnn(model, dataloaders, loss, optimizer, logger, args.save_freq, args.save_path, args.epochs, has_mask)
        else:
            train(model, dataloaders, loss, optimizer, logger, args.save_freq, args.save_path, args.epochs, has_mask)

main()

# TODOs
# Add logging
# Add validation
# Save model every 5-10 epochs (or every epoch)
# Fix training so that it segments mnist properly
