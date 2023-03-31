import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import datetime
from random import randrange
from agents.dqn import DQN
from agents.agent_dqn import AgentDQN
from agents.transition_table import TransitionTable
from agents.preproc import Preproc
from agents.dialog import Dialog

class NeuralQLearner(object):

    def __init__(self, agent_params, transition_params):

        self.agent_params = agent_params
        self.transition_params = transition_params

        self.agent_type = agent_params["agent_type"]
        self.frame_pooling_style = agent_params["frame_pooling_style"]
        self.log_dir = agent_params["log_dir"]
        self.show_extra_plots = agent_params["show_extra_plots"]
        self.gpu = agent_params["gpu"]
        self.device = torch.device("cuda" if self.gpu >= 0 else "cpu")
        self.n_actions = agent_params["n_actions"]
        self.hist_len = agent_params["hist_len"]
        self.downsample_w = agent_params["downsample_w"]
        self.downsample_h = agent_params["downsample_h"]
        self.extra_info_size = agent_params["extra_info_size"]
        self.max_reward = agent_params["max_reward"]
        self.min_reward = agent_params["min_reward"]
        self.ep_start = agent_params["ep_start"]
        self.ep = self.ep_start
        self.ep_end = agent_params["ep_end"]
        self.ep_endt = agent_params["ep_endt"]
        self.discount = agent_params["discount"]
        self.learn_start = agent_params["learn_start"]
        self.update_freq = agent_params["update_freq"]
        self.n_replay = agent_params["n_replay"]
        self.minibatch_size = agent_params["minibatch_size"]
        self.target_refresh_steps = agent_params["target_refresh_steps"]
        self.show_graphs = agent_params["show_graphs"] = True
        self.graph_save_freq = agent_params["graph_save_freq"]

        self.mc_return_required = agent_params["mc_return_required"]

        self.in_channels = self.hist_len

        self.numSteps = 0

        # For inserting complete episodes into the experience replay cache
        if self.mc_return_required:
            self.current_episode = []
            
        self.lastState = None
        self.lastAction = None
        self.lastTerminal = False

        if self.agent_type == 'dqn':
            self.agent = AgentDQN(self, agent_params)
            self.bestq = np.zeros((1), dtype=np.float32)
            
        else:
            sys.exit('Agent type (' + self.agent_type + ') not recognised!')


        self.preproc = Preproc(self.agent_params, self.downsample_w, self.downsample_h)
        self.transitions = TransitionTable(self.transition_params)

        self.episode_score = 0
        self.episode_score_clipped = 0
        self.moving_average_score = 0
        self.moving_average_score_clipped = 0
        self.moving_average_score_mom = 0.98
        self.moving_average_score_updates = 0

        self.q_values_plot = Dialog()
        self.score_plot = Dialog()
        self.clipped_score_plot = Dialog()


    def add_episode_to_cache(self):

        IDX_STATE = 0
        IDX_ACTION = 1
        IDX_EXTRINSIC_REWARD = 2
        IDX_TERMINAL = 3

        ep_length = len(self.current_episode)
        ret = np.zeros((ep_length), dtype=np.float32)

        i = ep_length - 1
        ret[i] = self.current_episode[i][IDX_EXTRINSIC_REWARD]

        i = ep_length - 2
        while i >= 0:
            ret[i] = self.current_episode[i][IDX_EXTRINSIC_REWARD] + self.discount * ret[i + 1]
            i -= 1

        # Add episode to the cache
        i = 0
        while i < ep_length:
            self.transitions.add(self.current_episode[i][IDX_STATE], self.current_episode[i][IDX_ACTION], self.current_episode[i][IDX_EXTRINSIC_REWARD], ret[i], self.current_episode[i][IDX_TERMINAL])
            i = i + 1

        self.current_episode = []


    def handle_game_over(self):

        print(str(datetime.datetime.now()) + ', neural_q_learner received termination signal, self.episode_score = ' + str(self.episode_score))

        self.moving_average_score = self.moving_average_score_mom * self.moving_average_score + (1.0 - self.moving_average_score_mom) * self.episode_score
        self.moving_average_score_clipped = self.moving_average_score_mom * self.moving_average_score_clipped + (1.0 - self.moving_average_score_mom) * self.episode_score_clipped

        self.moving_average_score_updates = self.moving_average_score_updates + 1

        zero_debiased_score = self.moving_average_score / (1.0 - self.moving_average_score_mom ** self.moving_average_score_updates)
        zero_debiased_score_clipped = self.moving_average_score_clipped / (1.0 - self.moving_average_score_mom ** self.moving_average_score_updates)

        self.score_plot.add_data_point("movingAverageScore", self.numSteps, [zero_debiased_score], False, self.show_graphs)
        self.clipped_score_plot.add_data_point("movingAverageClippedScore", self.numSteps, [zero_debiased_score_clipped], False, self.show_graphs)

        if self.show_extra_plots:
            self.agent.update_plots()

        self.episode_score = 0
        self.episode_score_clipped = 0


    def perceive(self, reward, rawstate, extra_info, terminal, game_over):

        #rawstate = torch.from_numpy(rawstate)
        #state = self.preproc.forward(rawstate)

        state = cv2.resize(rawstate, (self.downsample_w, self.downsample_h), interpolation=cv2.INTER_AREA)
        
        state = torch.from_numpy(state).float()
        extra_info = torch.from_numpy(extra_info).float()
        
        # Update the unclipped, undiscounted total reward (i.e. the game score)
        self.episode_score += reward

        self.transitions.add_recent_state(state, extra_info, terminal)
        curState, curExtra = self.transitions.get_recent()
        curState = curState.reshape(1, self.hist_len, self.downsample_w, self.downsample_h).to(self.device)

        # Clip the reward
        reward = np.minimum(reward, self.max_reward)
        reward = np.maximum(reward, self.min_reward)

        self.episode_score_clipped += reward

        # Store transition s, a, r, s'
        if self.lastState is not None:
        
            if self.mc_return_required:
                self.current_episode.append((self.lastState, self.lastAction, reward, self.lastTerminal))
            else:
                self.transitions.add(self.lastState, self.lastAction, reward, 0.0, self.lastTerminal)

        if game_over:
            self.handle_game_over()

        # Necessary to process episode once lastTerminal == True so that each experience in the cache has a full return.
        if self.lastTerminal and self.mc_return_required:
            self.add_episode_to_cache()

        # Select action
        actionIndex = 0
        if not terminal:
            actionIndex = self.eGreedy(curState, curExtra)

        self.q_values_plot.add_data_point("bestq", self.numSteps, self.bestq, True, self.show_graphs)
        #if True: # self.ale.isUpdatingScreenImage():
        #    self.q_values_plot.update_image('self.ep = ' + str(self.ep) + ', replay num ent = ' + str(self.transitions.size()))

        if self.numSteps % self.graph_save_freq == 0:
            self.q_values_plot.save_image(self.log_dir)
            self.score_plot.save_image(self.log_dir)
            self.clipped_score_plot.save_image(self.log_dir)

            if self.show_extra_plots:
                self.agent.save_plots(self.log_dir)

        self.numSteps += 1

        # Do some Q-learning updates
        if self.numSteps > self.learn_start and self.numSteps % self.update_freq == 0:
            for i in range(0, self.n_replay):
                self.agent.learn()

        self.lastState = state.clone()
        self.lastAction = actionIndex
        self.lastTerminal = terminal

        if self.numSteps % self.target_refresh_steps == 0:
            self.agent.refresh_target()

        return actionIndex


    def eGreedy(self, state, extra_info):

        self.ep = self.ep_end + np.maximum(0, (self.ep_start - self.ep_end) * (self.ep_endt - np.maximum(0, self.numSteps - self.learn_start)) / self.ep_endt)

        # Can uncomment the below for better efficiency while self.ep == 1
        #if self.ep == 1:
        #    return randrange(self.n_actions)
        
        a = self.agent.greedy(state, extra_info)

        # Epsilon greedy
        if np.random.uniform() < self.ep:
            return randrange(self.n_actions)
        else:
            return a
