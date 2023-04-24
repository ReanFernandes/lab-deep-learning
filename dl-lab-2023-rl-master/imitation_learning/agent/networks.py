import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.n_classes = n_classes
        self.history_length = history_length
        self.conv1 = nn.Conv2d(in_channels=n_classes, out_channels=32, kernel_size=12)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p = 0.2)

        self.fc1 = nn.Linear(in_features= self._conv_to_linear_size(), out_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=32)
        self.bn5 = nn.BatchNorm1d(num_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=3)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  #layer 1
        x = self.relu(self.bn2(self.conv2(x)))  #layer 2
        x = self.relu(self.bn3(self.conv3(x)))  #layer 3
        x = self.flatten(x)                     #flatten
        x = self.relu(self.bn4(self.fc1(x)))    #fully connected layer 1
        x = self.dropout(x)                     #dropout for regularization
        x = self.relu(self.bn5(self.fc2(x)))    #fully connected layer 2
        x = self.fc3(x)                         #fully connected layer 3
        x = self.softmax(x)                     #softmax

        return x

    def _conv_to_linear_size(self):
        # calculates the size of the linear layer because i feel like this is easier than hand calculating
        x = torch.rand(self.history_length,96, 96, self.n_classes)
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output  = self.flatten(output)
        return output.shape[1]

