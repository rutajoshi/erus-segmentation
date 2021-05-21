import argparse, sys
import torch
from torchvision.datasets import MNIST as Dataset
import torch.optim as optim
from unet.unet import UNet

HAS_CUDA = torch.cuda.is_available()

CONFIG_CIFAR = { # config for cifar10
    "image_size": 34,
    "padding": 1,
    "units": [3, 32, 32,
              64, 64, 128, 128, 64, 64, 64, 64, 64],
    "train_data": torchvision.datasets.CIFAR10('../cifar', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Pad(1)
    ])),
    "test_data": torchvision.datasets.CIFAR10('../cifar', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Pad(1)
    ]))
}

CONFIG_MNIST = { # config for MNIST
    "image_size": 28,
    "padding": 0,
    "units": [1, 32, 32,
              64, 64, 128, 128, 64, 64, 64],
    "train_data": torchvision.datasets.MNIST('../mnist', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])),
    "test_data": torchvision.datasets.MNIST('../mnist', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ]))
}

CONFIG = CONFIG_MNIST

def get_cifar_dataloader(batch_size=128):
    # TODO: make a dataloader for CIFAR
    return

def get_mnist_dataloader(batch_size=128):
    # Get train and test data
    train_data = CONFIG["train_data"]
    test_data = CONFIG["test_data"]

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def load_dataset(dataset_name, split, batch_size):
    train_loader, test_loader = None, None
    if (dataset_name == 'mnist'):
        train_loader, test_loader = get_mnist_dataloader(batch_size)
    elif (dataset_name == 'cifar'):
        train_loader, test_loader = get_cifar_dataloader(batch_size)
    return train_loader, test_loader

def load_model(model_name, dataset_name):
    # Default
    model = UNet(in_channels=1, out_channels=1)
    if (model_name == 'unet'):
        if (dataset_name == 'cifar'):
            model = UNet(in_channels=3, out_channels=1)
    elif (model_name == 'gnn'):
        agg = GCNNAgg((4,4), agg_method='max')
        gnn = BaseGNN(5, None, CONFIG["units"], agg, torch.nn.functional.relu)
        seg_head = GNNSeg(gnn, 10)
        model = seg_head
    return model

def load_loss(loss_name):
    loss = torch.nn.CrossEntropyLoss()
    return loss

def load_optimizer(params, optimizer_name, learning_rate):
    optimizer = optim.Adam(params, lr=learning_rate)
    return optimizer

def train(model, dataloader, loss_function, optimizer):
    """
    By default: train forever
    """
    epoch = 0
    step = 0
    loss_train = []
    while True:
        print("Starting epoch " + str(epoch))
        for i, data in enumerate(dataloader):
            step += 1
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                y_pred = model(x)
                loss = loss_function(y_pred, y_true)

                if (step % 100 == 0):
                    print("Loss at step " + str(step) + " = " + str(loss.detach().cpu().numpy()))

                loss_train.append(loss.item())
                loss.backward()
                optimizer.step()
        epoch += 1
        if (epoch % 10 == 0):
            print("On epoch: " + str(epoch))

def inference(model, dataset):
    return

# Train a model given the model name, dataset, loss function, and other parameters
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', help='Model type', type=str, default='unet')
    parser.add_argument('--dataset', help='Dataset name', type=str, default='mnist')
    parser.add_argument('--loss', help='Loss function (CE/FL/DL)', type=str, default='CE')
    parser.add_argument('--optimizer', help='SGD/Adam', type=str, default='SGD')
    parser.add_argument('--learning_rate', help='Learning rate', type=int, default=0.01)
    parser.add_argument('--split', help='Train/valid/test', type=str, default='train')
    parser.add_argument('--inference', help='boolean inference', type=bool, default=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--device', help='cuda if available', type=str, default='cpu')
    args=parser.parse_args()

    # Pick device
    device = torch.device("cpu" if not HAS_CUDA else args.device)

    # Load dataset
    train_loader, test_loader = load_dataset(args.dataset, args.split, args.batch_size)
    # Load model
    model = load_model(args.model, args.dataset)
    # Load loss function
    loss = load_loss(args.loss)
    # Load optimizer
    optimizer = load_optimizer(model.parameters(), args.optimizer, args.learning_rate)

    # Train or do inference
    if (args.inference):
        if (args.split == "train"):
            inference(model, train_loader)
        else if (args.split == "test"):
            inference(model, test_loader)
    else:
        train(model, train_loader, loss, optimizer)

main()

# TODOs
# Add logging
# Save model every 5-10 epochs (or every epoch)
# Fix training so that it segments mnist properly
