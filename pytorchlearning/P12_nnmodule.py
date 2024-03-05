import torch
from torch import nn


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


test = TestModule()
x = torch.tensor(1.0)
outout = test(x)
print(outout)


