import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader

# input = torch.tensor([[1, 0.5],
#               [-1,3]])
#
# input = torch.reshape(input, (-1,1,2,2))
#
# print(input.shape)
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
        # 非线性激活
        self.relu1 = ReLU(inplace=False)
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


test = TestModule()


writer = SummaryWriter("test_logs")

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


# print(test(input))
