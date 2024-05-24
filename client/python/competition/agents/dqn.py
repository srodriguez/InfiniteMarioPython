import torch
import torch.nn as nn
import numpy as np
from prettytable import PrettyTable

class DQN(nn.Module):

    def __init__(self, gpu, in_channels, extra_latent_size, num_actions):
        super(DQN, self).__init__()

        self.projection_size = 8
        
        # Michael note: See 'ZLevelMapElementGeneralization' in LevelScene.java (ch.idsia.mario.engine)
        # Michael note: Also see Sprite.java (ch.idsia.mario.engine.sprites)

        # 16 is brick (simple). Also used for bricks with hidden items to prevent cheating.
        # 20 is angry flower pot or cannot. I guess it's most similar to a hard obstacle, but maybe more like 'dangerous obstacle'?
        # 21 is question brick -- most similar to hard obstacle

        ### Current List of observations tags - will have to change in the future to be dynamically generated via agent perception
        self.listObservations = [
            2,3,4,5,6,7,8,9,10,12,13,14,15,16,20,21,25,34,245,246,
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 # Dummy values for day/night
        ]
        
        """     
        self.dictionaryList ={
            'enemies':[2,3,4,5,6,7,8,9,10,12,13],
            'hard_obstacles':[16,21,246],
            'platform_obstacles':[245],
            'dangerous_obstacles':[20],
            'powerups':[14, 15],
            'coins':[34],
            'fireballs':[25]
        }
        """

        self.device = torch.device("cuda" if gpu >= 0 else "cpu")
        
        self.conv1 = nn.Conv2d(in_channels=in_channels*len(self.listObservations), out_channels=in_channels*self.projection_size, kernel_size=1, stride=1).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=in_channels*self.projection_size, out_channels=64, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2).to(self.device)
        
        self.fc1_a = nn.Linear(in_features=4*4*64 + extra_latent_size, out_features=256).to(self.device)
        self.fc1_v = nn.Linear(in_features=4*4*64 + extra_latent_size, out_features=256).to(self.device)

        self.fc2_a = nn.Linear(in_features=256, out_features=num_actions).to(self.device)
        self.fc2_v = nn.Linear(in_features=256, out_features=num_actions).to(self.device)

        # In the Nature paper the biases aren't zeroed, although it's probably good practice to zero them.
        # self.conv1.bias.data.fill_(0.0)
        # self.conv2.bias.data.fill_(0.0)
        # self.conv3.bias.data.fill_(0.0)
        # self.fc1.bias.data.fill_(0.0)
        # self.fc2.bias.data.fill_(0.0)

        self.relu = nn.ReLU().to(self.device)
        self.count_parameters()


    def forward(self, x, extra_info):

        x = self.binaryWorldMaker(x).float().to(self.device)

        # Calculate categories
        x = self.conv1(x)#.permute((0, 2, 3, 1))
        #x = nn.functional.softmax(x, dim=3).permute((0, 3, 1, 2))
        x = nn.functional.softmax(x, dim=1)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        
        x = torch.cat((x, extra_info.view(x.size(0), -1).to(self.device)), dim=1)
        
        a = self.relu(self.fc1_a(x))
        a = self.relu(self.fc2_a(a))

        v = self.relu(self.fc1_v(x))
        v = self.relu(self.fc2_v(v))

        x = v + a - a.mean(a.dim() - 1, keepdim=True)

        return x
        
        
    def binaryWorldMaker(self, state):
    
        binary_reps = []
        for obs_id in self.listObservations:
            binary_reps.append(torch.eq(state, obs_id))

        return torch.cat(binary_reps, dim=1)
        

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

