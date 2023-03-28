import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):

    def __init__(self, gpu, in_channels, extra_latent_size, num_actions):
        super(DQN, self).__init__()

        self.num_categories = 3 # Enemies, obstacles, powerups
        
        #self.dictionaryList ={'enemies':[2,9,25,20],'obstacles':[-10,-11],'powerups':[16,21]}
        self.dictionaryList ={'enemies':[2,9,25,20],'obstacles':[246,245],'powerups':[16,21]}
    
        self.device = torch.device("cuda" if gpu >= 0 else "cpu")
        
        #self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4).to(self.device)
        self.conv1 = nn.Conv2d(in_channels=in_channels*self.num_categories, out_channels=32, kernel_size=4, stride=2).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2).to(self.device)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1).to(self.device)
        
        #self.fc1 = nn.Linear(in_features=7*7*64, out_features=512).to(self.device)
        self.fc1 = nn.Linear(in_features=4*4*16 + extra_latent_size, out_features=128).to(self.device)
        self.fc2 = nn.Linear(in_features=128, out_features=num_actions).to(self.device)

        # In the Nature paper the biases aren't zeroed, although it's probably good practice to zero them.
        # self.conv1.bias.data.fill_(0.0)
        # self.conv2.bias.data.fill_(0.0)
        # self.conv3.bias.data.fill_(0.0)
        # self.fc1.bias.data.fill_(0.0)
        # self.fc2.bias.data.fill_(0.0)

        self.relu = nn.ReLU().to(self.device)


    def forward(self, x, extra_info):
    
        x = self.binaryWorldMaker(x, self.dictionaryList).float().to(self.device)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        #x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        
        x = torch.cat((x, extra_info.view(x.size(0), -1).to(self.device)), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
        
    def binaryWorldMaker(self, state, dictionary):
    
        enemy_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('enemies')], dim=0), dim = 0)
        obstacle_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('obstacles')], dim=0), dim = 0)
        powerup_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('powerups')], dim=0), dim = 0)
        
        return torch.cat((enemy_rep, obstacle_rep, powerup_rep), dim=1)
        
    
    def binaryWorldMaker_old(self, state, dictionary):
    
        state = state.cpu().numpy()
        n_batch = state.shape[0]
        n_channels_in = state.shape[1]
        
        # TODO: Un-hardcode the 22x22
        output = np.zeros((n_batch, n_channels_in * self.num_categories, 22, 22), dtype=bool)
        
        for batch in range(n_batch):
            for channel in range(n_channels_in):
            
                img = state[batch][channel]

                # Enemies
                counter = 0
                for i in img:
                    for e in dictionary.get('enemies'):
                        idList = [v for v, val in enumerate(i) if val == e]
                        for location in idList:
                            output[batch][channel * self.num_categories][counter][location] = 1
                    counter += 1

                # Obstacles
                counter = 0
                for i in img:
                    for e in dictionary.get('obstacles'):
                        idList = [v for v, val in enumerate(i) if val == e]
                        for location in idList:
                            output[batch][channel * self.num_categories + 1][counter][location] = 1
                    counter += 1

                # Powerups
                counter = 0
                for i in img:
                    for e in dictionary.get('powerups'):
                        idList = [v for v, val in enumerate(i) if val == e]
                        for location in idList:
                            output[batch][channel * self.num_categories + 2][counter][location] = 1
                    counter += 1
        
        return torch.from_numpy(output)

