from PIL import Image
from torchvision import transforms

img_path = r"dataset_new\train\ants_image\0013035.jpg"
img = Image.open(img_path)
print(img)

# 转化为tensor类型

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

