import torchvision
from torch.utils.tensorboard import SummaryWriter
import os

log_path = r"P10"
for file in os.listdir(log_path):
    file_path = os.path.join(log_path, file)
    os.remove(file_path)

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False, transform=dataset_transform, download=True)


writer = SummaryWriter("P10")

for i in range(10):
    img, target = test_set[i]
    print("img %s target is %s" %(i, test_set.classes[target]))
    writer.add_image("From_P10", img, i)

writer.close()

