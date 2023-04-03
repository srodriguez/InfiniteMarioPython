import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):

    def __init__(self, gpu, in_channels, extra_latent_size, num_actions):
        super(DQN, self).__init__()

        self.unique_ids = None

        self.num_categories = 7 # enemies, hard obstacles, platform obstacles, dangerous obstacles, powerups, coins, fireballs
        
        # Michael note: See 'ZLevelMapElementGeneralization' in LevelScene.java (ch.idsia.mario.engine)
        # Michael note: Also see Sprite.java (ch.idsia.mario.engine.sprites)

        # 16 is brick (simple). Also used for bricks with hidden items to prevent cheating.
        # 20 is angry flower pot or cannot. I guess it's most similar to a hard obstacle, but maybe more like 'dangerous obstacle'?
        # 21 is question brick -- most similar to hard obstacle

        self.dictionaryList ={
            'enemies':[2,3,4,5,6,7,8,9,10,12,13],
            'hard_obstacles':[16,21,246],
            'platform_obstacles':[245],
            'dangerous_obstacles':[20],
            'powerups':[14, 15],
            'coins':[34],
            'fireballs':[25]
        }

        #self.dictionaryList ={'enemies':[2,3,4,5,6,7,8,9,10,12],'obstacles':[245,246],'powerups':[16,21]}
    
        self.device = torch.device("cuda" if gpu >= 0 else "cpu")
        
        #self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4).to(self.device)
        self.conv1 = nn.Conv2d(in_channels=in_channels*self.num_categories, out_channels=64, kernel_size=4, stride=2).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2).to(self.device)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1).to(self.device)
        
        #self.fc1 = nn.Linear(in_features=7*7*64, out_features=512).to(self.device)
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


    def forward(self, x, extra_info):
    
        if self.unique_ids is None:
            self.unique_ids = x.unique()
        else:
            new_ids = torch.cat((self.unique_ids, x.unique())).unique()
            if new_ids.size()[0] > self.unique_ids.size()[0]:
                #print("Old categories:")
                #print(self.unique_ids)
                #print("New categories:")
                #print(new_ids)
                #print(x[0][3])
                #input()
                self.unique_ids = new_ids

        x = self.binaryWorldMaker(x, self.dictionaryList).float().to(self.device)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        #x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        
        x = torch.cat((x, extra_info.view(x.size(0), -1).to(self.device)), dim=1)
        
        a = self.relu(self.fc1_a(x))
        a = self.relu(self.fc2_a(a))

        v = self.relu(self.fc1_v(x))
        v = self.relu(self.fc2_v(v))

        x = v + a - a.mean(a.dim() - 1, keepdim=True)

        return x
        
        
    def binaryWorldMaker(self, state, dictionary):
    
        enemy_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('enemies')], dim=0), dim = 0)
        hard_obstacle_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('hard_obstacles')], dim=0), dim = 0)
        platform_obstacle_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('platform_obstacles')], dim=0), dim = 0)
        dangerous_obstacle_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('dangerous_obstacles')], dim=0), dim = 0)
        powerup_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('powerups')], dim=0), dim = 0)
        coin_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('coins')], dim=0), dim = 0)
        fireball_rep = torch.any(torch.stack([torch.eq(state, aelem).logical_or_(torch.eq(state, aelem)) for aelem in dictionary.get('fireballs')], dim=0), dim = 0)

        return torch.cat((enemy_rep, hard_obstacle_rep, platform_obstacle_rep, dangerous_obstacle_rep, powerup_rep, coin_rep, fireball_rep), dim=1)
        
    
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

