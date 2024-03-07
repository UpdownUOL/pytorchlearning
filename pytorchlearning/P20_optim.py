import torch
import torchvision.datasets
from torch.nn import Conv2d,MaxPool2d,Linear,Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

test_loader = DataLoader(dataset=dataset, batch_size=64, drop_last=True)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = torch.nn.Sequential(
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


loss = torch.nn.CrossEntropyLoss()
writer = SummaryWriter("test_logs")

test = TestModule()

optim = torch.optim.SGD(test.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in test_loader:
        imgs, targets = data
        outputs = test(imgs)
        result_loss = loss(outputs, targets)
        # 设置梯度为0
        optim.zero_grad()
        # 反向传播的梯度
        result_loss.backward()
        # 对参数进行调优
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)