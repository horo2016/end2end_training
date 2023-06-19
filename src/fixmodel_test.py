import torch
import torch.nn as nn

# 输入是一个N=1，C=3，H=70，W=320的向量

conv_layers = nn.Sequential(
            
            nn.Conv2d(1, 3, 5, stride=2),
            nn.ELU(),
            #nn.Conv2d(8, 12, 5, stride=2),
            #nn.ELU(),
            nn.Conv2d(3, 8, 5, stride=2),
            nn.ELU(),
            #nn.Conv2d(12, 24, 3),
            #nn.ELU(),
            nn.Conv2d(8, 8, 3),
            nn.Dropout(0.5)
    )
input = torch.randn(1, 1, 200, 200)
output = conv_layers(input)
print(output.shape)#torch.Size([1, 64, 2, 33])