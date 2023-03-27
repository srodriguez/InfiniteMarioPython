import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):

    def __init__(self, gpu, in_channels, num_actions):
        super(DQN, self).__init__()

        self.device = torch.device("cuda" if gpu >= 0 else "cpu")

        #self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4).to(self.device)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=1).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1).to(self.device)

        #self.fc1 = nn.Linear(in_features=7*7*64, out_features=512).to(self.device)
        self.fc1 = nn.Linear(in_features=6*6*64, out_features=512).to(self.device)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions).to(self.device)

        # In the Nature paper the biases aren't zeroed, although it's probably good practice to zero them.
        # self.conv1.bias.data.fill_(0.0)
        # self.conv2.bias.data.fill_(0.0)
        # self.conv3.bias.data.fill_(0.0)
        # self.fc1.bias.data.fill_(0.0)
        # self.fc2.bias.data.fill_(0.0)

        self.relu = nn.ReLU().to(self.device)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x