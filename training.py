
import time
import sys
import subprocess
import json
import random
import training_helpers
import numpy as np
import torch

# import custom structures (like MESSAGE, ACTION)
from custom_structures import *
from state import *
from data_pokemon import *

EPOCHS = 30
MAX_GAME_LEN = 400 #max length is 200 but if you u-turn every turn you move twice per turn
BATCH_SIZE = 100
torch.set_default_dtype(torch.float64)
class VPGBuffer:
    """
    Buffer that stores trajectories from a batch of games
    """
    def _init_(self, size = MAX_GAME_LEN*BATCH_SIZE, gamma=0.99, lam=0.95):
        self.state_buffer = {}
        self.action_buffer = np.zeros(dtype=int) #won't overflow since max number is >> possible length
        self.adv_buffer = np.zeros(dtype=np.float32)
        self.rew_buffer = np.zeros(dtype=np.float32)
        self.rtg_buffer = np.zeros(dtype=np.float32) #the rewards-to-go
        self.val_buffer = np.zeros(dtype=np.float32) #save in np because we recompute value a bunch anyway
        self.logp_buffer = torch.zeros(size).float() #logp value
        self.gamma = gamma, self.lam = lam 
        self.total_tuples = 0 #so we know where to cut off vectors above for updates
        self.ptr_start = 0 #an index of the start of the trajectory currently being put in memory
        self.ptr = 0 #an index of the next tuple to be put in the buffer

    def end_traj(self):
        '''
        Call at the end of an episode to update buffers with rewards to go and advantage
        '''
        path_slice = slice(self.ptr_start, self.ptr)
        rews = self.rew_buffer[path_slice]
        vals = self.val_buffer[path_size]

        #compute GAE advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buffer[path_slice] = training_helpers.discount_cumsum(deltas, self.gamma * self.lam)

        #compute rewards-to-go
        self.ret_buffer[path_slice] = training_helpers.discount_cumsum(rews, self.gamma)[:-1]

        self.ptr_start = self.ptr
        return

    def get(self):
        '''
        Call after a batch to normalize the advantages and return the data needed for network updates
        '''
        # nornalize advantage values to mean 0 std 1
        adv_mean = np.mean(self.adv_buffer)
        adv_std = np.std(self.adv_buffer)
        self.adv_buffer = (self.adv_buffer - adv_mean) / adv_std
        return [self.state_buffer, self.act_buffer, self.adv_buffer, 
                self.rtg_buffer, self.logp_buffer]




#create a class instance for our learning agent needs a policy architecture and value function architecture
#idea for ablation: train the value function on a fully observed state space instead of partially observed for the purpose of computing advantage

torch.manual_seed(42)
np.random,seed(42)

train_v_iters = 80

#initialize the network class here

for i in EPOCHS:
    #collect experience
    #initialize the buffer object. use a new buffer object for each batch since we want it reset after network updates anyway
    buffer_obj = VPGBuffer()
    #run BATCH_SIZE games and save a list of lists of tuples see line 29
    for j in BATCH_SIZE:
        #for every action taken
        #input state, action, value estimate, rewards, logp(action) into the buffer
        #increment self.ptr

    #trajectory ends
    buffer_obj.end_traj()
    state, _, adv, rtg, logp = buffer_obj.get() 

    #take a policy gradient step
    #"loss" = 1/batch_size * sum logp_t * adv_t
    policy_loss = 1/BATCH_SIZE * logp * adv

    #backprop step on the policy "loss"

    #do a few iterations of value function. #for this must recompute that value given the state at each iteration
    for _ in range(train_v_iters):
        #compute value of s in a forward pass

        #loss = mse(val(s_t) - rtg_t)
        #gradient update on the loss

    #save the policy and value function every few epochs






