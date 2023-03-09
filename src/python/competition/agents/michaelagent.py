import numpy
import random
import sys
import os
import torch
import torch.nn as nn
from neural_q_learner import NeuralQLearner
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 1, 2009 2:46:34 AM$"

from marioagent import MarioAgent

class MichaelAgent(MarioAgent):
    """ In fact the Python twin of the
        corresponding Java ForwardAgent.
    """
    action = None
    actionStr = None
    KEY_JUMP = 3
    KEY_SPEED = 4
    levelScene = None
    mayMarioJump = None
    isMarioOnGround = None
    marioFloats = None
    enemiesFloats = None
    isEpisodeOver = False

    trueJumpCounter = 0;
    trueSpeedCounter = 0;


    def reset(self):
        self.isEpisodeOver = False
        self.trueJumpCounter = 0;
        self.trueSpeedCounter = 0;
        self.cumulativeReward = 0
        self.lastReward = 0
        

    def __init__(self):
        """Constructor"""
        self.use_gpu = False
        self.trueJumpCounter = 0
        self.trueSpeedCounter = 0
        self.action = numpy.zeros(5, int)
        self.action[1] = 1
        self.actionStr = ""

        self.actions = []
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 2):
                    for l in range(0, 2):
                        for m in range(0, 2):
                            self.actions.append([i, j, k, l, m])

        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        #self.fc1 = nn.Linear(in_features=484, out_features=100).to(self.device)
        #self.fc2 = nn.Linear(in_features=100, out_features=len(self.actions)).to(self.device)

        self.reset()

        #####################
        #### AGENT SETUP ####
        #####################

        print 'Started setting up the agent...'
        agent_params = {}

        agent_params["agent_type"] = "dqn"

        agent_params["frame_pooling_style"] = "max_pool" # color_averaging, max_pool

        # Optimizer settings
        agent_params["optimizer"] = "rms_prop" # "rms_prop", "adam"

        agent_params["rms_prop_lr"] = 0.00025
        agent_params["rms_prop_alpha"] = 0.95
        agent_params["rms_prop_eps"] = 0.1 / 32.0 # Note: The way epsilon is expressed in the Nature paper (= 0.1) doesn't follow convention because in the code they don't divide the loss by the batch size.

        agent_params["adam_lr"] = 0.0000625
        agent_params["adam_eps"] = 0.00015
        agent_params["adam_beta1"] = 0.9
        agent_params["adam_beta2"] = 0.999

        agent_params["log_dir"] = os.path.dirname(os.path.realpath(__file__))
        agent_params["log_dir"] = agent_params["log_dir"] + '/results/' + agent_params["agent_type"] + "/"
        if not os.path.exists(agent_params["log_dir"]):
            os.makedirs(agent_params["log_dir"])
    
        agent_params["show_extra_plots"] = False
        agent_params["gpu"] = 0 if self.use_gpu else -1
        agent_params["n_actions"] = len(self.actions)
        agent_params["use_rgb_for_raw_state"] = False
        agent_params["hist_len"] = 4
        agent_params["downsample_w"] = 22 # 84
        agent_params["downsample_h"] = 22 # 84
        agent_params["max_reward"] = 1.0 # Use float("inf") for no clipping
        agent_params["min_reward"] = -1.0 # Use float("-inf") for no clipping
        agent_params["ep_start"] = 0.5 # 1
        agent_params["ep_end"] = 0.1
        agent_params["ep_endt"] = 1000000
        agent_params["discount"] = 0.99
        agent_params["learn_start"] = 800 # 50000
        agent_params["update_freq"] = 4
        agent_params["n_replay"] = 1
        agent_params["minibatch_size"] = 32
        agent_params["target_refresh_steps"] = 10000
        agent_params["show_graphs"] = True
        agent_params["graph_save_freq"] = 1000

        # For training methods that require the Monte Carlo return for each episode, set the below to True.
        agent_params["mc_return_required"] = False

        transition_params = {}
        transition_params["agent_params"] = agent_params
        transition_params["replay_size"] = 1000000
        transition_params["hist_spacing"] = 1
        transition_params["bufferSize"] = 512

        self.q_learner = NeuralQLearner(agent_params, transition_params)


    def getAction(self):
        """ Possible analysis of current observation and sending an action back
        """
        a_idx = self.q_learner.perceive(self.lastReward, self.levelScene, self.isEpisodeOver, self.isEpisodeOver)
        self.action = self.actions[a_idx]
        return self.action
        

    def integrateObservation(self, obs):
        """This method stores the observation inside the agent"""
        if (len(obs) != 6):
            self.isEpisodeOver = True
            self.lastReward = 0 # TODO: Fix
            self.cumulativeReward = 0
        else:
            self.isEpisodeOver = False
            self.lastReward = obs[2][0] - self.cumulativeReward
            self.cumulativeReward = obs[2][0]
            self.mayMarioJump, self.isMarioOnGround, self.marioFloats, self.enemiesFloats, self.levelScene, dummy = obs


    def printLevelScene(self):
        ret = ""
        for x in range(22):
            tmpData = ""
            for y in range(22):
                tmpData += self.mapElToStr(self.levelScene[x][y]);
            ret += "\n%s" % tmpData;
        print ret


    def mapElToStr(self, el):
        """maps element of levelScene to str representation"""
        s = "";
        if  (el == 0):
            s = "##"
        s += "#MM#" if (el == 95) else str(el)
        while (len(s) < 4):
            s += "#";
        return s + " "


    def printObs(self):
        """for debug"""
        print repr(self.observation)

