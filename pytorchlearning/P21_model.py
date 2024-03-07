import torch
import torchvision
# 使用现有模型，修改适配

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)


# 增加一个module
vgg16_true.add_module(name="add_linear", module=torch.nn.Linear(1000, 10))
