import torch
import torchvision.datasets
from torch.nn import Conv2d,MaxPool2d,Linear,Flatten
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)


test_loader = DataLoader(dataset=dataset, batch_size=1, drop_last=True)


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 线性化 前面是入参size，后面是结果的size
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

        self.model1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output



loss = nn.CrossEntropyLoss()
writer = SummaryWriter("test_logs")

test = TestModule()

for data in test_loader:
    imgs, targets = data
    outputs = test(imgs)
    print(outputs)
    print(targets)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print(result_loss)