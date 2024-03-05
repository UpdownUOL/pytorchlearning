# kernel_size
# stride 步长
# padding
# dilation
# 池化就是一堆里面选择一个


import torch
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=dataset_transform,
                                       download=True)

test_loader = DataLoader(dataset=dataset, batch_size=64)

# input = torch.tensor([[1, 2, 3, 4, 5],
#                       [0, 4, 6, 7, 8],
#                       [3, 4, 4, 7, 2],
#                       [1, 3, 5, 7, 5],
#                       [1, 3, 5, 7, 5]],dtype=torch.float32)
#
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
#
# input = torch.reshape(input, (-1, 1, 5, 5))
# kernel = torch.reshape(kernel, (-1, 1, 3, 3))


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 最大池化
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


writer = SummaryWriter("test_logs")

test = TestModule()
# output = test(input)
# print(output)

step = 0
for data in test_loader:
    imgs, targets = data
    output = test(imgs)

    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()