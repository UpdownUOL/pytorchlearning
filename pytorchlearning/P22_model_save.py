import torch
import torchvision

# 模型结构加参数

vgg16 = torchvision.models.vgg16(pretrained=False)

# 模型参数
torch.save(vgg16.state_dict(), "vgg_test.pth")