import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from random import randrange
from dqn import DQN

class AgentDQN(object):

    def __init__(self, manager, agent_params):

        self.manager = manager

        self.device = torch.device("cuda" if self.manager.gpu >= 0 else "cpu")

        self.discount = agent_params["discount"]

        self.network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions)
        self.target_network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions)
        self.target_network.load_state_dict(self.network.state_dict())
        
        if agent_params["optimizer"] == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=agent_params["adam_lr"], betas=(agent_params["adam_beta1"], agent_params["adam_beta2"]), eps=agent_params["adam_eps"])
        elif agent_params["optimizer"] == 'rms_prop':
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=agent_params["rms_prop_lr"], alpha=agent_params["rms_prop_alpha"], eps=agent_params["rms_prop_eps"], weight_decay=0, momentum=0, centered=True)
        else:
            sys.exit('Optimizer type (' + agent_params["optimizer"] + ') not recognised!')
 
 
    def learn(self):

        assert self.manager.transitions.size() > self.manager.minibatch_size, 'Not enough transitions stored to learn'

        s, a, r, _, s2, term = self.manager.transitions.sample(self.manager.minibatch_size)

        r = torch.from_numpy(r).float().to(self.device)
        term = torch.from_numpy(term).float().to(self.device)
        a_tens = torch.from_numpy(a).to(self.device).unsqueeze(1).long()

        q_tp1 = self.target_network.forward(s2).detach()

        # Calculate q-values at time t
        q_values = self.network.forward(s).gather(1, a_tens).squeeze()

        value_tp1, _ = q_tp1.max(1)

        # An alternative is to calculate the greedy action first, then gather the Q-values.
        # This makes it easier to implemented Double DQN.
        # _, greedy_act = q_tp1.max(1)
        # greedy_act = greedy_act.unsqueeze(1)
        # value_tp1 = q_tp1.gather(1, greedy_act).squeeze()
        
        target_overall = torch.ones_like(term).sub(term).mul(self.discount).mul(value_tp1).add(r)

        error = q_values - target_overall

        # Huber loss
        error.clamp_(-1.0, 1.0)

        error.div_(self.manager.minibatch_size)

        self.optimizer.zero_grad()
        q_values.backward(error.data)
        self.optimizer.step()


    def refresh_target(self):

        self.target_network.load_state_dict(self.network.state_dict())


    def greedy(self, state):

        # Turn single state into minibatch.  Needed for convolutional nets.
        assert state.dim() >= 3, 'Input must be at least 3D'

        q = self.network.forward(state).cpu().detach().squeeze()
        q = q.numpy()

        maxq = q[0]
        besta = [0]

        # Evaluate all other actions (with random tie-breaking)
        for a in range(1, self.manager.n_actions):

            if q[a] > maxq:
                besta = [a]
                maxq = q[a]

            elif q[a] == maxq:
                besta.append(a)

        r = randrange(len(besta))
        action_selected = besta[r]

        self.manager.bestq[0] = q[action_selected]

        return action_selected
