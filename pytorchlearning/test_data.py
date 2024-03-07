import torch
import torchvision.datasets
from torch.nn import Conv2d,MaxPool2d,Linear,Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)

print(dataset.classes[7])