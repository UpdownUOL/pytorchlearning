import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=dataset_transform,
                                       download=True)

test_loader = DataLoader(dataset=dataset, batch_size=64)


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("test_logs")

test = TestModule()

step = 0
for data in test_loader:
    imgs, targets = data
    output = test(imgs)
    # batch_size channel 长，宽
    print(imgs.shape)
    print(output.shape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()