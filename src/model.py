"""
CNN model architecture.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import torch.nn as nn
import torch.nn.functional as F


class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.

        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)

        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 1 * 18, out_features=50),
            #nn.ELU(),
            #nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """Forward pass."""
        x= input.size(0)
        
        input = input.view(input.size(0), 1, 66, 200)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        output = output.squeeze(-1)
        return output
class myNetworkFromNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize my model.

          model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 3, strides: 4x4, activation: ELU
            Convolution: 5x5, filter: 8, strides: 4x4, activation: ELU
            Convolution: 5x5, filter: 8,  activation: ELU
           
            Drop out (0.5)
           
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)

        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(myNetworkFromNvidia, self).__init__()
        self.conv_layers = nn.Sequential(

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
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1* 8 * 45 *45, out_features=5),
            nn.ELU(),
            #nn.Linear(in_features=100, out_features=50),
            #nn.ELU(),
            #nn.Linear(in_features=50, out_features=10),
            #nn.ELU(),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, input):
        """Forward pass."""
        x= input.size(0)
        #print(x)
        input = input.view(input.size(0), 1, 200, 200)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        output = output.squeeze(-1)
        return output


class LeNet(nn.Module):
    """LeNet architecture."""

    def __init__(self):
        """Initialization."""
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass."""
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
