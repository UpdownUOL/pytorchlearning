import torchvision
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# test data set
test_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=dataset_transform,
                                        download=True)

# batch_size 是打包，把多少张数据放在一起
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

#测试数据集第一张样本
img, target = test_set[0]
print(img.shape)
print(target)
print(test_set.classes[target])

for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)

