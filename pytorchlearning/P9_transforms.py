from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os

img_path = r"test_pic/2020_01_06_05_58_IMG_5045.JPG"
img = Image.open(img_path)


log_path = r"logs"
for file in os.listdir(log_path):
    file_path = os.path.join(log_path, file)
    os.remove(file_path)

writer = SummaryWriter("logs")

# 转化为tensor类型

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("To_tensor", tensor_img)

# 归一化 Normalize
print(tensor_img[0][0][0])
# output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("To_Normalize", img_norm)


# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
print(img_resize.size())
writer.add_image("To_Resize", img_resize)


# Compose 等比缩放
print(img.size)
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
print(img_resize_2.size())
writer.add_image("To_Resize_2", img_resize_2)


writer.close()


