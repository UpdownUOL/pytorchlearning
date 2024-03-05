import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 3, 4, 5],
                      [0, 4, 6, 7, 8],
                      [3, 4, 4, 7, 2],
                      [1, 3, 5, 7, 5],
                      [1, 3, 5, 7, 5]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (-1, 1, 5, 5))
kernel = torch.reshape(kernel, (-1, 1, 3, 3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)


output2 = F.conv2d(input, kernel, stride=2)
print(output2)


output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)


output = F.max_pool2d(input, kernel_size = 3, ceil_mode =True)
print(output)


output2 = F.max_pool2d(input, kernel_size = 2, ceil_mode =True)
print(output2)


output3 = F.max_pool2d(input, kernel_size = 3, ceil_mode =False)
print(output3)
