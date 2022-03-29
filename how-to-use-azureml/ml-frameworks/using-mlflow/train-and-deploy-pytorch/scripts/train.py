# Copyright (c) 2017, PyTorch Team
# All rights reserved
# Licensed under BSD 3-Clause License.

# This example is based on PyTorch MNIST example:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

import mlflow
import mlflow.pytorch
from mlflow.utils.environment import _mlflow_conda_env
import warnings
import cloudpickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Added the view for reshaping score requests
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # Use MLflow logging
            mlflow.log_metric("epoch_loss", loss.item())


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\n")
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # Use MLflow logging
    mlflow.log_metric("average_loss", test_loss)


class Args(object):
    pass


# Training settings
args = Args()
setattr(args, 'batch_size', 64)
setattr(args, 'test_batch_size', 1000)
setattr(args, 'epochs', 3)  # Higher number for better convergence
setattr(args, 'lr', 0.01)
setattr(args, 'momentum', 0.5)
setattr(args, 'no_cuda', True)
setattr(args, 'seed', 1)
setattr(args, 'log_interval', 10)
setattr(args, 'save_model', True)

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# Use Azure Open Datasets for MNIST dataset
datasets.MNIST.mirrors = [
    "https://azureopendatastorage.azurefd.net/mnist/"
]
datasets.MNIST.resources = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
]
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def driver():
    warnings.filterwarnings("ignore")
    # Dependencies for deploying the model
    pytorch_index = "https://download.pytorch.org/whl/"
    pytorch_version = "cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl"
    deps = [
        "cloudpickle=={}".format(cloudpickle.__version__),
        pytorch_index + pytorch_version,
        "torchvision=={}".format(torchvision.__version__),
        "Pillow=={}".format("6.0.0")
    ]
    with mlflow.start_run() as run:
        model = Net().to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
        # Log model to run history using MLflow
        if args.save_model:
            model_env = _mlflow_conda_env(additional_pip_deps=deps)
            mlflow.pytorch.log_model(model, "model", conda_env=model_env)
    return run


if __name__ == "__main__":
    driver()
