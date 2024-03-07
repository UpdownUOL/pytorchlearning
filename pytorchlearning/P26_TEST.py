import torch
import torchvision.transforms
from PIL import Image
from P23_model import TestModule

dataset = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)


device = torch.device("cuda")

image_path = "img/img_2.png"
image = Image.open(image_path)
image = image.convert("RGB")

print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()
                                            ])

image = transform(image)

print(image)

test_model = torch.load("test.pth")

image = torch.reshape(image,(1,3,32,32))
image = image.to(device)

test_model.eval()
with torch.no_grad():
    output = test_model(image)

output_index = output.argmax(1)
print(output.argmax(1))


print(dataset.classes[output_index])