import torch
import torch.nn as nn
import numpy as np
from random import randrange

class TransitionTable(object):

    def __init__(self, transition_params):

        self.transition_params = transition_params

        self.agent_params = transition_params["agent_params"]
        self.gpu = self.agent_params["gpu"]
        self.hist_len = self.agent_params["hist_len"]
        self.discount = self.agent_params["discount"]
        self.downsample_w = self.agent_params["downsample_w"]
        self.downsample_h = self.agent_params["downsample_h"]
        self.extra_info_size = self.agent_params["extra_info_size"]
        self.n_step_n = self.agent_params["n_step_n"]
        self.replay_size = transition_params["replay_size"]
        self.hist_spacing = transition_params["hist_spacing"]
        self.bufferSize = transition_params["bufferSize"]

        self.zeroFrames = True
        self.recentMemSize = self.hist_spacing * self.hist_len
        self.numEntries = 0
        self.insertIndex = 0
        self.buf_ind = self.bufferSize + 1 # To ensure the buffer is always refilled initially

        self.histIndices = []

        for i in range(0, self.hist_len):
            self.histIndices.append(i * self.hist_spacing)

        self.s = torch.empty(self.replay_size, self.downsample_w, self.downsample_h, dtype=torch.uint8).zero_()
        self.extra_info = torch.empty(self.replay_size, self.extra_info_size, dtype=torch.float).zero_()
        self.a = np.zeros((self.replay_size), dtype=np.int32)
        self.r = np.zeros((self.replay_size), dtype=np.float32)
        self.ret = np.zeros((self.replay_size), dtype=np.float32)
        self.ret_partial = np.zeros((self.replay_size), dtype=np.float32)
        self.t = np.zeros((self.replay_size), dtype=np.int32)
        self.game_won = np.zeros((self.replay_size), dtype=np.int32)
        self.steps_until_term = np.zeros((self.replay_size), dtype=np.int32)

        self.recent_s = []
        self.recent_a = []
        self.recent_t = []
        self.recent_extra_info = []
        
        self.buf_a = np.zeros((self.bufferSize), dtype=np.int32)
        self.buf_r = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_ret = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_ret_partial = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_term = np.zeros((self.bufferSize), dtype=np.int32)
        self.buf_term_under_n = np.zeros((self.bufferSize), dtype=np.float32)
        self.buf_game_won = np.zeros((self.bufferSize), dtype=np.int32)
        self.buf_s = torch.empty(self.bufferSize, self.hist_len, self.downsample_w, self.downsample_h, dtype=torch.uint8).zero_()
        self.buf_s_plus_n = torch.empty(self.bufferSize, self.hist_len, self.downsample_w, self.downsample_h, dtype=torch.uint8).zero_()
        self.buf_extra_info = torch.empty(self.bufferSize, self.hist_len, self.extra_info_size, dtype=torch.float).zero_()
        self.buf_extra_info_plus_n = torch.empty(self.bufferSize, self.hist_len, self.extra_info_size, dtype=torch.float).zero_()
        
        if self.gpu >= 0:
            self.gpu_s  = self.buf_s.float().cuda()
            self.gpu_s_plus_n = self.buf_s_plus_n.float().cuda()
            self.gpu_extra_info = self.buf_extra_info.float().cuda()
            self.gpu_extra_info_plus_n = self.buf_extra_info.float().cuda()
            

    def size(self):
        return self.numEntries


    def fill_buffer(self):

        assert self.numEntries > self.bufferSize, 'Not enough transitions stored to learn'

        # clear CPU buffers
        self.buf_ind = 0

        for buf_ind in range(0, self.bufferSize):
            s, extra_info, a, r, ret, ret_partial, s_plus_n, extra_info_plus_n, term, term_under_n, game_won = self.sample_one()
            self.buf_s[buf_ind].copy_(s)
            self.buf_extra_info[buf_ind].copy_(extra_info)
            self.buf_a[buf_ind] = a
            self.buf_r[buf_ind] = r
            self.buf_ret[buf_ind] = ret
            self.buf_ret_partial[buf_ind] = ret_partial
            self.buf_s_plus_n[buf_ind].copy_(s_plus_n)
            self.buf_extra_info_plus_n[buf_ind].copy_(extra_info_plus_n)
            self.buf_term[buf_ind] = term
            self.buf_term_under_n[buf_ind] = term_under_n
            self.buf_game_won[buf_ind] = game_won

        #self.buf_s = self.buf_s.float().div(255)
        #self.buf_s_plus_n = self.buf_s_plus_n.float().div(255)

        if self.gpu >= 0:
            self.gpu_s.copy_(self.buf_s)
            self.gpu_s_plus_n.copy_(self.buf_s_plus_n)
            self.gpu_extra_info.copy_(self.buf_extra_info)
            self.gpu_extra_info_plus_n.copy_(self.buf_extra_info_plus_n)
            

    def sample_one(self):

        assert self.numEntries > 1, 'Experience cache is empty'

        valid = False

        while not valid:
            # Lua comment: start at 1 because of previous action
            # Michael note: To be honest, I'm not sure why we can't start at 0, but it's not a big deal
            # Michael note: Also, the below logic means the last few indices in the experience cache won't ever be selected
            index = randrange(1, self.numEntries - self.recentMemSize)
            if self.t[index + self.recentMemSize - 1] == 0:
                valid = True

        return self.get(index)


    def sample(self, batch_size):

        assert batch_size < self.bufferSize, 'Batch size must be less than the buffer size'

        if self.buf_ind + batch_size > self.bufferSize:
            self.fill_buffer()

        index = self.buf_ind
        index2 = index + batch_size

        self.buf_ind = self.buf_ind + batch_size

        if self.gpu >=0:
            return self.gpu_s[index:index2], self.gpu_extra_info[index:index2], self.buf_a[index:index2], self.buf_r[index:index2], self.buf_ret[index:index2], self.buf_ret_partial[index:index2], self.gpu_s_plus_n[index:index2], self.gpu_extra_info_plus_n[index:index2], self.buf_term[index:index2], self.buf_term_under_n[index:index2], self.buf_game_won[index:index2]
        else:
            return self.buf_s[index:index2], self.buf_extra_info[index:index2], self.buf_a[index:index2], self.buf_r[index:index2], self.buf_ret[index:index2], self.buf_ret_partial[index:index2], self.buf_s_plus_n[index:index2], self.buf_extra_info_plus_n[index:index2], self.buf_term[index:index2], self.buf_term_under_n[index:index2], self.buf_game_won[index:index2]


    def concatFrames(self, index, use_recent):

        if use_recent:
            s, t, extra_info = self.recent_s, self.recent_t, self.recent_extra_info
        else:
            s, t, extra_info = self.s, self.t, self.extra_info

        fullstate = torch.empty(self.hist_len, self.downsample_w, self.downsample_h, dtype=torch.uint8).zero_()
        fullextra = torch.empty(self.hist_len, self.extra_info_size, dtype=torch.float).zero_()

        # Zero out frames from all but the most recent episode.
        # This is achieved by looking *back* in time from the current frame (index + self.histIndices[self.hist_len - 1])
        # until a terminal state is found. Frames before the terminal state then get zeroed.
        # The index logic is a bit tricky... The index where self.t[index] == 1 is actually the first state of the new episode.
        zero_out = False
        episode_start = self.hist_len - 1

        for i in range(self.hist_len - 2, -1, -1):

            if not zero_out:

                for j in range(index + self.histIndices[i], index + self.histIndices[i + 1]):

                    if t[self.wrap_index(j, use_recent)] == 1:
                        zero_out = True
                        break

            if zero_out:
                fullstate[i].zero_()
                fullextra[i].zero_()
            else:
                episode_start = i

        # Could get rid of this since it's never called. Just here to match up with the Lua code (where it is also never called).
        if not self.zeroFrames:
            episode_start = 0

        # Copy frames from the current episode.
        for i in range(episode_start, self.hist_len):
            fullstate[i].copy_(s[self.wrap_index(index + self.histIndices[i], use_recent)])
            fullextra[i].copy_(extra_info[self.wrap_index(index + self.histIndices[i], use_recent)])
            
        return fullstate, fullextra


    def get_recent(self):

        # Assumes that the most recent state has been added, but the action has not
        fullstate, fullextra = self.concatFrames(0, True)
        #return fullstate.float().div(255), fullextra
        return fullstate, fullextra


    def wrap_index(self, index, use_recent=False):

        if use_recent:
            return index

        if self.numEntries == 0:
            return index

        while index < 0:
            index += self.numEntries

        while index >= self.numEntries:
            index -= self.numEntries

        return index


    def get(self, index):

        s, extra_info = self.concatFrames(index, False)
        s_plus_n, extra_info_plus_n = self.concatFrames(self.wrap_index(index + self.n_step_n), False)
        ar_index = index + self.recentMemSize - 1

        term_under_n = 0
        if (self.steps_until_term[ar_index] - 1) < self.n_step_n:
            term_under_n = 1

        return s, extra_info, self.a[ar_index], self.r[ar_index], self.ret[ar_index], self.ret_partial[ar_index], s_plus_n, extra_info_plus_n, self.t[ar_index + 1], term_under_n, self.game_won[ar_index + 1]


    def add(self, s, extra_info, a, r, ret, ret_partial, term, game_won, steps_until_term):

        assert s is not None, 'State cannot be nil'
        assert a is not None, 'Action cannot be nil'
        assert r is not None, 'Reward cannot be nil'

        term_stored_value = 0
        if term:
            term_stored_value = 1

        game_won_stored_value = 0
        if game_won:
            game_won_stored_value = 1

        # Overwrite (s, a, r, t) at insertIndex
        self.s[self.insertIndex] = s.byte()
        self.extra_info[self.insertIndex] = extra_info.clone()
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        self.ret[self.insertIndex] = ret
        self.ret_partial[self.insertIndex] = ret_partial
        self.t[self.insertIndex] = term_stored_value
        self.game_won[self.insertIndex] = game_won_stored_value
        self.steps_until_term[self.insertIndex] = steps_until_term

        # Increment until at full capacity
        if self.numEntries < self.replay_size:
            self.numEntries += 1

        # Always insert at next index, then wrap around
        self.insertIndex += 1

        # Overwrite oldest experience once at capacity
        if self.insertIndex >= self.replay_size:
            self.insertIndex = 0


    def add_recent_state(self, s, extra_info, term):

        s = s.byte()

        if len(self.recent_s) == 0:
            for i in range(0, self.recentMemSize):
                self.recent_s.append(s.clone().zero_())
                self.recent_t.append(1)
                self.recent_extra_info.append(extra_info.clone().zero_())
                
        self.recent_s.append(s)
        self.recent_extra_info.append(extra_info)

        if term:
            self.recent_t.append(1)
        else:
            self.recent_t.append(0)

        # Keep recentMemSize states.
        if len(self.recent_s) > self.recentMemSize:
            del self.recent_s[0]
            del self.recent_t[0]
            del self.recent_extra_info[0]
