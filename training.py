'''
Skeleton code for VPG. Partly based on  https://github.com/openai/spinningup/tree/master/spinup/algos/ppo
'''


import time
import sys
import subprocess
import json
import random
import training_helpers
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import game_coordinator
from agents import *
import csv

# import custom structures (like MESSAGE, ACTION)
from custom_structures import *
from state import *
from data_pokemon import *
import neural_net
from neural_net import DeePikachu0

EPOCHS = 100
MAX_GAME_LEN = 4000 #max length is 200 but if you u-turn every turn you move twice per turn
BATCH_SIZE = 32 # 8 #100
ACTION_SPACE_SIZE = 10 #4 moves and 6 switches

def action_to_int(action):
    #moves are 0 through 3, switches and teamspecs are 4 through 9
    if(action['id'] == 'move'):
        return int(action['movespec'])-1
    elif(action['id'] == 'team'):
        return int(action['teamspec'])+3
    #it must be a switch otherwise
    return int(action['switchspec'])+3

def int_to_action(x, teamprev = False):
    #opposite of action_to_int    
    if(x < 4):
        return {'id':'move', 'movespec': str(x+1)}
    elif teamprev:
        return {'id':'team', 'teamspec': str(x-3)}
    else:
        return {'id':'switch', 'switchspec': str(x-3)}


#torch.set_default_dtype(torch.float64)
class VPGBuffer:
    """
    Buffer that stores trajectories from a batch of games
    """
    def _init_(self, size = MAX_GAME_LEN*BATCH_SIZE, gamma=0.99, lam=0.95):
        self.state = copy.deepcopy(default_state)

        self.buffer_size = size
        self.state_buffer = create_2D_state(self.buffer_size)
        self.state2_buffer = create_2D_state(self.buffer_size)
        self.action_buffer = np.zeros(self.buffer_size, dtype=int) #won't overflow since max number is >> possible length
        self.adv_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.rew_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.rtg_buffer = np.zeros(self.buffer_size, dtype=np.float32) #the rewards-to-go
        self.val_buffer = np.zeros(self.buffer_size, dtype=np.float32) #save in np because we recompute value a bunch anyway
        self.logp_buffer = np.zeros(self.buffer_size, dtype=np.float32) #logp value
        self.valid_actions_buffer = np.zeros((self.buffer_size, ACTION_SPACE_SIZE)) #stores what actions were valid at that time point as a 1hot
        self.gamma = gamma
        self.lam = lam 
        self.total_tuples = 0 #so we know where to cut off vectors above for updates
        self.ptr_start = 0 #an index of the start of the trajectory currently being put in memory
        self.ptr = 0 #an index of the next tuple to be put in the buffer
        self.done_buffer = np.zeros(self.buffer_size, dtype = int)

    def end_traj(self):
        '''
        Call at the end of an episode to update buffers with rewards to go and advantage
        '''
        path_slice = slice(self.ptr_start, self.ptr)
        rews = np.append(self.rew_buffer[path_slice], 0) #0 added so the sizing works out
        vals = np.append(self.val_buffer[path_slice], 0)

        #add in state2 observastion
        if(self.ptr == 0 and self.total_tuples > 0):
            #if 0 pointer save at the end (you looped around the buffer length)
            self.state2_buffer = self.recurse_store_state(self.state2_buffer, self.state, index = self.total_tuples-1)
            self.done_buffer[self.total_tuples-1] = 1
        else:
            #else update the state before the current one
            self.state2_buffer = self.recurse_store_state(self.state2_buffer, self.state, index = self.ptr-1)
            self.done_buffer[self.ptr-1] = 1

        #compute GAE advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buffer[path_slice] = training_helpers.discount_cumsum(deltas, self.gamma * self.lam)

        #compute rewards-to-go
        self.rtg_buffer[path_slice] = training_helpers.discount_cumsum(rews, self.gamma)[:-1]

        self.state = copy.deepcopy(default_state)
        self.ptr_start = self.ptr
        return

    def recurse_cut_state(self, state_buffer):
        '''
        stores a state in buffer recursively
        '''
        for field in state_buffer:
            if (isinstance(state_buffer[field], dict)):
                state_buffer[field] = self.recurse_cut_state(state_buffer[field])
            else:
                state_buffer[field] = state_buffer[field][0:self.ptr]
        return state_buffer

    def get(self):
        '''
        Call after a batch to normalize the advantages and return the data needed for network updates
        '''
        # nornalize advantage values to mean 0 std 1
        adv_mean = np.mean(self.adv_buffer)
        adv_std = np.std(self.adv_buffer)
        self.adv_buffer = (self.adv_buffer - adv_mean) / adv_std
        return [self.recurse_cut_state(self.state_buffer), self.action_buffer[:self.ptr], self.adv_buffer[:self.ptr], 
                self.rtg_buffer[:self.ptr], self.logp_buffer[:self.ptr], self.valid_actions_buffer[:self.ptr]]


class LearningAgent(VPGBuffer, DefaultAgent):
    '''
    Consists of all VPGbuffer info and a network for selecting moves, along with all agent subclasses
    '''
    def __init__(self, id, name='Ash', size = MAX_GAME_LEN*BATCH_SIZE, gamma=0.99, lam=0.95, network=None):
        self.id = id
        self.name = name
        self.history = []
        self.state = copy.deepcopy(default_state)

        self.buffer_size = size
        self.state_buffer = create_2D_state(self.buffer_size)
        self.state2_buffer = create_2D_state(self.buffer_size)
        self.action_buffer = np.zeros(self.buffer_size, dtype=int) #won't overflow since max number is >> possible length
        self.adv_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.rew_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.rtg_buffer = np.zeros(self.buffer_size, dtype=np.float32) #the rewards-to-go
        self.val_buffer = np.zeros(self.buffer_size, dtype=np.float32) #save in np because we recompute value a bunch anyway
        self.logp_buffer = np.zeros(self.buffer_size, dtype=np.float32)#logp value
        self.valid_actions_buffer = np.zeros((self.buffer_size, ACTION_SPACE_SIZE)) #stores what actions were valid at that time point as a 1hot
        self.gamma = gamma
        self.lam = lam 
        self.total_tuples = 0 #so we know where to cut off vectors above for updates
        self.ptr_start = 0 #an index of the start of the trajectory currently being put in memory
        self.ptr = 0 #an index of the next tuple to be put in the buffer
        self.done_buffer = np.zeros(self.buffer_size, dtype = int)

        self.network = network
        self.wins = 0
        self.warmup = False
        self.minibatch_size = 100
        self.evalmode = False

    def recurse_store_state(self, state_buffer, state, index):
        '''
        stores a state in buffer recursively
        '''
        for field in state:
            if (isinstance(state[field], dict)):
                state_buffer[field] = self.recurse_store_state(state_buffer[field], state[field], index)
            else:
                state_buffer[field][index] = state[field]
        return state_buffer

    def construct_np_state_from_python_state(self, np_state, state):
        '''
        stores a state in buffer recursively
        '''
        for field in state:
            if (isinstance(state[field], dict)):
                np_state[field] = self.construct_np_state_from_python_state(np_state[field], state[field])
            else:
                np_state[field][0] = state[field]
        return np_state


    def store_in_buffer(self, state, action, value, logp, valid_actions):
        '''
        Stores everything in the buffer and increments the pointer
        '''

        #leave rewards as all 0 then impute later in the win function

        self.state_buffer = self.recurse_store_state(self.state_buffer, self.state, self.ptr)

        #storing the previous time-points next state
        #unless its the first turn of the game store state2 at previous time 
        if(self.ptr != self.ptr_start):
            if(self.ptr == 0 and self.total_tuples > 0):
                #if 0 pointer save at the end (you looped around the buffer length)
                self.state2_buffer = self.recurse_store_state(self.state2_buffer, self.state, index = self.total_tuples-1)
            else:
                #else update the state before the current one
                self.state2_buffer = self.recurse_store_state(self.state2_buffer, self.state, index = self.ptr-1)

        self.action_buffer[self.ptr] = action_to_int(action)

        self.val_buffer[self.ptr] = value

        self.logp_buffer[self.ptr] = logp

        #self.valid_actions[self.ptr] 
        for option in valid_actions:
            self.valid_actions_buffer[self.ptr, action_to_int(option)] = 1


        self.ptr+=1
        if(self.ptr == self.buffer_size):
            self.ptr = 0
        self.total_tuples = max(self.ptr, self.total_tuples) #if full stay full else increment with pointer
        return

    def won_game(self):
        '''
        sets this time's reward to 1 in the buffer
        '''
        if(self.ptr == 0 and total_tuples > 0):
            self.rew_buffer[self.total_tuples-1] = 1 
        else:
            self.rew_buffer[self.ptr-1] = 1 
        self.wins += 1

    def process_request(self, request):
        '''
        Same as the typical request handling for agents but uses the network to sample decisions
        and writes game information to the buffer using store_in_buffer
        '''
        self.request_update(request.message)
        message = request.message['request_dict']

        #save the state in the buffer

        #first get our valid action space
        valid_actions = get_valid_actions(self.state, message)
        

        if(self.network == None):
            if (valid_actions == []):
                raise ValueError("no valid actions")
                action = copy.deepcopy(ACTION['default'])
            else:
                action = random.choice(valid_actions)
            value = 0
            logp = np.log(1/min(1, len(valid_actions)))
        else:

            with torch.no_grad():
                if (valid_actions == []):
                    raise ValueError("no valid actions")
                    value = 0
                    action = copy.deepcopy(ACTION['default'])
                else:
                    is_teampreview = ('teamspec' in valid_actions[0])
                    np_state = create_2D_state(1) #initialize an empty np state to update
                    np_state = self.construct_np_state_from_python_state(np_state, self.state)
                    policy_tensor, value_tensor = self.network(np_state)
                    value = value_tensor[0]
                    if is_teampreview:
                        policy_tensor[0][0:4] *= 0
                    else:
                        for i in np.arange(10):
                            if int_to_action(i) not in valid_actions:
                                policy_tensor[0][i] *= 0

                    policy = policy_tensor.cpu().detach().numpy()[0]    
                    policy /= np.sum(policy)

                    #check if we're at teampreview and sample action accordingly. if at teampreview teamspec in first option 
                    if is_teampreview:
                        action = int_to_action(np.random.choice(np.arange(10), p=policy), teamprev = True)
                    else:
                        action = int_to_action(np.random.choice(np.arange(10), p=policy), teamprev = False)
                

                #save logpaction in buffer (not really needed since it gets recomputed)
                logp = np.log(1/min(1, len(valid_actions)))


        self.store_in_buffer(self.state, action, value, logp, valid_actions)
        
        return PlayerAction(self.id, action)

    def empty_buffer(self):
        self.state_buffer = create_2D_state(self.buffer_size)
        self.state2_buffer = create_2D_state(self.buffer_size)
        self.action_buffer = np.zeros(self.buffer_size, dtype=int) 
        self.adv_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.rew_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.rtg_buffer = np.zeros(self.buffer_size, dtype=np.float32) 
        self.val_buffer = np.zeros(self.buffer_size, dtype=np.float32) 
        self.logp_buffer = np.zeros(self.buffer_size, dtype=np.float32) 
        self.total_tuples = 0 
        self.ptr_start = 0 
        self.ptr = 0 
        self.valid_actions_buffer = np.zeros((self.buffer_size, ACTION_SPACE_SIZE)) 
        self.done_buffer = np.zeros(self.buffer_size, dtype = int)
        self.wins = 0

        return 

    def spit(self):
        """
        returns the class buffers
        """
        idxs = np.arange(self.total_tuples)
        return [self.recurse_index_state(copy.deepcopy(self.state_buffer), idxs), self.recurse_index_state(copy.deepcopy(self.state2_buffer), idxs), self.action_buffer[idxs], self.adv_buffer[idxs], 
                self.rtg_buffer[idxs], self.logp_buffer[idxs], self.valid_actions_buffer[idxs], self.rew_buffer[idxs], self.done_buffer[idxs]]


class SACAgent(LearningAgent):
    '''
    New class for Q learning
    '''
    def process_request(self, request):
        '''
        Uses a Q function instead of a policy. For SAC take exp of "policy"
        '''

        self.request_update(request.message)
        message = request.message['request_dict']
        #save the state in the buffer

        #first get our valid action space
        valid_actions = get_valid_actions(self.state, message)

        if(self.warmup or self.network == None):
            if (valid_actions == []):
                action = copy.deepcopy(ACTION['default'])
            else:
                action = random.choice(valid_actions)
            value = 0
            logp = np.log(1/min(1, len(valid_actions)))
        else:
            is_teampreview = ('teamspec' in valid_actions[0])
            np_state = create_2D_state(1) #initialize an empty np state to update
            np_state = self.construct_np_state_from_python_state(np_state, self.state)
            policy_tensor, _, value_tensor = self.network(np_state)
            #print(policy_tensor)
            value = value_tensor[0]  

            policy_tensor = torch.exp(policy_tensor)
            
            if is_teampreview:
                for i in np.arange(10):
                    if int_to_action(i, teamprev=True) not in valid_actions:
                        policy_tensor[0][i] *= 0
            else:
                for i in np.arange(10):
                    if int_to_action(i) not in valid_actions:
                        policy_tensor[0][i] *= 0
            

            policy = policy_tensor.cpu().detach().numpy()[0] 
            policy /= np.sum(policy)
            #print(policy)
            #check if we're at teampreview and sample action accordingly. if at teampreview teamspec in first option 
            if is_teampreview:
                if(self.evalmode):
                    action = int_to_action(np.argmax(policy), teamprev = True)
                else:
                    action = int_to_action(np.random.choice(np.arange(10), p=policy), teamprev = True)
            else:
                if(self.evalmode):
                    action = int_to_action(np.argmax(policy), teamprev = False)
                else:
                    action = int_to_action(np.random.choice(np.arange(10), p=policy), teamprev = False)

            #save logpaction in buffer (not really needed since it gets recomputed)
            logp = np.log(1/min(1, len(valid_actions)))


        self.store_in_buffer(self.state, action, value, logp, valid_actions)
        
        return PlayerAction(self.id, action)

    def recurse_index_state(self, state_buffer, idxs):
        '''
        stores a state in buffer recursively
        '''
        for field in state_buffer:
            if (isinstance(state_buffer[field], dict)):
                state_buffer[field] = self.recurse_index_state(state_buffer[field], idxs)
            else:
                state_buffer[field] = state_buffer[field][idxs]
        return state_buffer

    def get(self):
        '''
        Call after a batch to return a sample from the buffer
        '''
        #idxs = np.arange(0, self.total_tuples)
        idxs = np.random.choice(self.total_tuples, size=self.minibatch_size, replace=False)
        # nornalize advantage values to mean 0 std 1
        #adv_mean = np.mean(self.adv_buffer)
        #adv_std = np.std(self.adv_buffer)
        #self.adv_buffer = (self.adv_buffer - adv_mean) / adv_std
        return [self.recurse_index_state(copy.deepcopy(self.state_buffer), idxs), self.recurse_index_state(copy.deepcopy(self.state2_buffer), idxs), self.action_buffer[idxs], self.adv_buffer[idxs], 
                self.rtg_buffer[idxs], self.logp_buffer[idxs], self.valid_actions_buffer[idxs], self.rew_buffer[idxs], self.done_buffer[idxs]]

#create a class instance for our learning agent needs a policy architecture and value function architecture
#idea for ablation: train the value function on a fully observed state space instead of partially observed for the purpose of computing advantage

torch.manual_seed(42)
np.random.seed(42)

#initialize the network class here

#pseudocode:
'''
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
    #call clear history on the agents
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
'''

if __name__ == '__main__':
    '''
    Trains LearningAgent vs RandomAgent
    usage:
    python3 training.py mode=train
    '''
    state_embedding_settings = {
        'pokemon' :     {'embed_dim' : 32, 'dict_size' : neural_net.MAX_TOK_POKEMON},
        'type' :        {'embed_dim' : 8, 'dict_size' : neural_net.MAX_TOK_TYPE},
        'move' :        {'embed_dim' : 8, 'dict_size' : neural_net.MAX_TOK_MOVE},
        'move_type' :   {'embed_dim' : 8, 'dict_size' : neural_net.MAX_TOK_MOVE_TYPE},
        'ability' :     {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_ABILITY},
        'item' :        {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_ITEM},
        'condition' :   {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_CONDITION},
        'weather' :     {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_WEATHER},
        'alive' :       {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_ALIVE},
        'disabled' :    {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_DISABLED},
        'spikes' :      {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_SPIKES},
        'toxicspikes' : {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_TOXSPIKES},
        'fieldeffect' : {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_FIELD},
    }

    d_player = 16
    d_opp = 16
    d_field = 16

    p1 = SACAgent(id='p1', name='Red', size = MAX_GAME_LEN*BATCH_SIZE, gamma=0.99, lam=0.95, 
        network=DeePikachu0(
            state_embedding_settings, 
            d_player=d_player, 
            d_opp=d_opp, 
            d_field=d_field, 
            dropout=0.0, 
            attention=True))
            
    p2 = RandomAgent(id='p2', name='Blue')

    # lr = 0.0004 # from SAC paper appendix
    lr = 0.001 
    weight_decay = 1e-4
    optimizer = optim.Adam(p1.network.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize value target at current network
    v_target_net = copy.deepcopy(p1.network)

    mse_loss = nn.MSELoss(reduction='mean')

    train_update_iters = 50

    alpha = 0.005
    warm_up = 3 #number of epochs playing randomly
    minibatch_size = 100
    p1.minibatch_size = minibatch_size
    max_winrate = 0
    win_array = []
    train_win_array = []

    #handle command line input whether to train or test
    if(len(sys.argv)>1 and sys.argv[1] == 'test'):
        p1.network.load_state_dict(torch.load('output/network__0.pth', map_location=torch.device('cpu')))
        p1.network.eval()
        p1.evalmode = True
        for i in range(0, 2):
            winner = game_coordinator.run_learning_episode(p1, p2)
            p1.end_traj()
            p1.clear_history()
            p2.clear_history()
        '''
        #debug stuff
        for i in range(0, 10):
            print(p1.recurse_index_state(copy.deepcopy(p1.state_buffer), i)['player']['active'])
            print(p1.action_buffer[i])
            print(p1.done_buffer[i])
            print(p1.rew_buffer[i])
            print(p1.valid_actions_buffer[i])
            print()
            #print(p1.recurse_index_state(copy.deepcopy(p1.state2_buffer), i)['opponent']['active'])
            print()
            #print(p1.recurse_index_state(copy.deepcopy(p1.state_buffer), i)['opponent']['active'])
        print(p1.ptr)
        '''
        print(p1.wins)
    else:
        for i in range(EPOCHS):
            p1.evalmode=False
            p1.network.train()
            print('Epoch: ', i)
            starttime = time.time()

            if(i > warm_up):
                p1.warmup = False
            else:
                p1.warmup = True


            for j in range(BATCH_SIZE): 
                game_coordinator.run_learning_episode(p1, p2)
                p1.end_traj()
                p1.clear_history()
                p2.clear_history()
                


            endttime = time.time()

            if(p1.total_tuples > p1.minibatch_size):
                for _ in range(train_update_iters):

                    '''
                    Soft Actor critic 
                    (Discrete, so no policy net)
                    '''

                    # Random sample from buffer (experience replay)
                    states, states2, actions, advs, rtgs, logps, valid_actions, rews, dones = p1.get()

                    actions = torch.tensor(actions, dtype=torch.long)
                    advs = torch.tensor(advs, dtype=torch.float)
                    rtgs = torch.tensor(rtgs, dtype=torch.float)
                    logps = torch.tensor(logps, dtype=torch.float)
                    valid_actions = torch.tensor(valid_actions, dtype=torch.long)
                    rews = torch.tensor(rews, dtype=torch.float)
                    dones = torch.tensor(dones, dtype=torch.long)

                    total_traj_len = actions.shape[0]
                    # compute supervised learning targets
                    with torch.no_grad():
                        
                        # value function target for s'
                        v_target_net.eval()
                        _, _, v_tensor_fixed = v_target_net(states2)
                        v_target_net.train()

                        # q function for s, a pairs
                        p1.network.eval()
                        q_tensor_A_fixed, q_tensor_B_fixed, _ = p1.network(states)
                        p1.network.train()


                        # q function regression target
                        q_target = rews + p1.gamma * (1 - dones) * v_tensor_fixed
                        
                        # v function regression target (min over both q heads:)
                        # 1
                        valid_q_A = torch.mul(valid_actions, torch.exp(q_tensor_A_fixed))
                        valid_policy_A = valid_q_A / valid_q_A.sum(dim=1, keepdim=True)

                        actions_tilde = torch.distributions.Categorical(probs=valid_policy_A).sample()

                        v_target_A = q_tensor_A_fixed[torch.arange(total_traj_len), actions_tilde] \
                            - alpha * torch.log(valid_policy_A[torch.arange(total_traj_len), actions_tilde])

                        # 2
                        v_target_B = q_tensor_B_fixed[torch.arange(total_traj_len), actions_tilde] \
                            - alpha * torch.log(valid_policy_A[torch.arange(total_traj_len), actions_tilde])
                        
                        # min
                        v_target = torch.min(torch.stack([v_target_A, v_target_B], dim=1), dim=1)[0]

                    #print(v_target)
                    #print(v_target)
                    # run updates on the networks
                    p1.network.train()

                    # Q step A
                    optimizer.zero_grad()
                    q_tensor_A, _, _ = p1.network(states)
                    q_action_taken_A = q_tensor_A[torch.arange(total_traj_len), actions]

                    loss = mse_loss(q_action_taken_A, q_target)
                    loss.backward()
                    optimizer.step()  

                    print('Q step A: ', loss.detach().item(), end='\t')

                    # Q step B
                    optimizer.zero_grad()
                    _, q_tensor_B, _ = p1.network(states)
                    q_action_taken_B = q_tensor_B[torch.arange(
                        total_traj_len), actions]

                    loss = mse_loss(q_action_taken_B, q_target)
                    loss.backward()
                    optimizer.step()

                    print('Q step B: ', loss.detach().item(), end='\t')

                    # V step
                    optimizer.zero_grad()
                    _, _, value_tensor = p1.network(states)
                   
                    loss = mse_loss(value_tensor, v_target)
                    loss.backward()
                    optimizer.step()

                    print('V step: ', loss.detach().item(), end='\n')

                    # Update target network for value function using exponential moving average
                    
                    with torch.no_grad():
                                
                        polyak = 0.995 # (default in openai pseudocode)
                        for param, param_target in zip(p1.network.parameters(), v_target_net.parameters()):
                            param_target.data.copy_(polyak * param_target.data + (1 - polyak) * param.data)


            # End epoch
            win_rate = float(p1.wins) / float(BATCH_SIZE) 
            # total win rate over all games in batch
            print('Batch win rate: ' + str(win_rate))
            p1.wins = 0
            train_win_array.append(win_rate)
            #p1.empty_buffer()

            #do an eval epoch
            if (i % 3 == 0):
                p1.network.eval()
                
                # agent plays argmax of q function
                p1.evalmode=True
                p1.warmup = False

                p1wins, p2wins = 0, 0

                for j in range(BATCH_SIZE): 
                    game_coordinator.run_learning_episode(p1, p2)
                    p1.clear_history()
                    p2.clear_history()
                    p1.end_traj()

                p1.evalmode=False
                p1wins = p1.wins
                p2wins = BATCH_SIZE - p1wins

                p1winrate = float(p1wins) / float(BATCH_SIZE)
                p2winrate = 1 - p1winrate 
            
                max_test_winrate = max(p1winrate, max_winrate)

                print('\n[Epoch {:3d}: Evaluation]  \n'.format(i) )
                print('Player 1 | win rate : {0:.4f} |  '.format(p1winrate) + 'wins : {:4d}  '.format(p1wins) + int(50 * p1winrate) * '#')
                print('Player 2 | win rate : {0:.4f} |  '.format(p2winrate) + 'wins : {:4d}  '.format(p2wins) + int(50 * p2winrate) * '#')
                print()

                p1.wins=0
                if(p1winrate >= max_test_winrate):
                    torch.save(p1.network.state_dict(), 'output/network_'+'_'+str(i)+'.pth')
                win_array.append(p1winrate)

        with open('output/results' + '.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(win_array)

        with open('output/train_results' + '.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(train_win_array)



