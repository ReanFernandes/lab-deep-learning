import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length, output_classes, batch_size): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.history_length = history_length
        self.output_classes = output_classes
        self.num_channels = 1 * history_length   # greyscale * histroy length
        self.batch_size = batch_size
        
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self._conv_to_linear_size()
        self.fc1 = nn.Linear(self.linear_size, 512)
        self.fc2 = nn.Linear(512, self.output_classes)


    def forward(self, x):
        # make sure data has the correct dimensions
        # x = x.view(-1, self.num_channels, 96, 96)
        #check if data tensor is on device
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float().to(self.device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    

    def _conv_to_linear_size(self):
        # calculates the size of the linear layer because i feel like this is easier than hand calculating
        x = torch.randn(self.batch_size,self.num_channels,96,96)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        self.linear_size = x.shape[1]
        

