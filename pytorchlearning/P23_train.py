import torch
import torchvision.datasets
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from P23_model import TestModule

# 准备数据集
dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataset_test = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)

dataset_loader = DataLoader(dataset=dataset, batch_size=64, drop_last=True)
dataset_test_loader = DataLoader(dataset=dataset_test, batch_size=64, drop_last=True)

train_data_size = len(dataset)
test_data_size = len(dataset_test)

print("训练数据集:{}".format(train_data_size))
print("测试数据集:{}".format(test_data_size))

if __name__ == "__main__":

    # 准备网络模型
    test = TestModule()

    # 准备损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 准备优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)

    total_train_step = 0
    total_test_step = 0

    epoch = 3

    for i in range(epoch):
        test.train()
        print("第 %s 轮 训练" % i)
        for data in dataset_loader:
            imgs, targets = data
            outputs = test(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数 %s ,loss %s" % (total_train_step, loss.item()))

        # 设置测试
        test.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in dataset_test_loader:
                imgs, targets = data
                outputs = test(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()

                total_accuracy = total_accuracy + accuracy

        total_test_step = total_test_step + 1
        print("测试次数 %s" % total_test_step)
        print("整体 loss %s" % total_test_loss)
        print("整体 正确率 %s" % (int(total_accuracy) / test_data_size))
