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

# import custom structures (like MESSAGE, ACTION)
from custom_structures import *
from state import *
from data_pokemon import *
import neural_net
from neural_net import DeePikachu0

EPOCHS = 30
MAX_GAME_LEN = 4000 #max length is 200 but if you u-turn every turn you move twice per turn
BATCH_SIZE = 10 #100
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
        print(self.state)
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
            if (valid_actions == []):
                value = 0
                action = copy.deepcopy(ACTION['default'])
            else:
                is_teampreview = ('teamspec' in valid_actions[0])
                np_state = create_2D_state(1) #initialize an empty np state to update
                np_state = self.construct_np_state_from_python_state(np_state, self.state)
                policy_tensor, _, value_tensor = self.network(np_state)
                print(policy_tensor)
                value = value_tensor[0]  

                policy_tensor = torch.exp(policy_tensor)
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
        idxs = np.random.randint(0, self.total_tuples, size=self.minibatch_size)
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
        network=DeePikachu0(state_embedding_settings, d_player=d_player, d_opp=d_opp, d_field=d_field, dropout=0.5, softmax=False))
    p2 = RandomAgent(id='p2', name='Blue')

    optimizer = optim.Adam(p1.network.parameters(), lr=0.01, weight_decay=1e-4)
    value_loss_fun = nn.MSELoss(reduction='mean')

    train_update_iters = 5

    alpha = 0.05
    warm_up = 0 #number of epochs playing randomly
    minibatch_size = 200
    p1.minibatch_size = minibatch_size

    #handle command line input whether to train or test
    if(len(sys.argv)>1 and sys.argv[1] == 'test'):
        p1.network.load_state_dict(torch.load('network_14.pth'))
        p1.network.eval()
        p1.evalmode = True
        winner = game_coordinator.run_learning_episode(p1, p2)
        p1.clear_history()
        p2.clear_history()
        p1.end_traj()
        print(p1.wins)
    else:
        for i in range(EPOCHS):

            print('Epoch: ', i)
            starttime = time.time()

            if(i > warm_up):
                p1.warmup = False
            else:
                p1.warmup = True


            for j in range(BATCH_SIZE): 
                game_coordinator.run_learning_episode(p1, p2)
                p1.clear_history()
                p2.clear_history()
                p1.end_traj()


            endttime = time.time()

            if(p1.total_tuples > p1.minibatch_size):
                for _ in range(train_update_iters):
                    states, states2, actions, advs, rtgs, logps, valid_actions, rews, dones = p1.get()

                    actions = torch.tensor(actions, dtype=torch.long)
                    advs = torch.tensor(advs, dtype=torch.float)
                    rtgs = torch.tensor(rtgs, dtype=torch.float)
                    logps = torch.tensor(logps, dtype=torch.float)
                    valid_actions = torch.tensor(valid_actions, dtype=torch.long)
                    rews = torch.tensor(rews, dtype=torch.float)
                    dones = torch.tensor(dones, dtype=torch.long)

                    total_traj_len = actions.shape[0]

                    # Q step
                    print('Q step')
                    optimizer.zero_grad()

                    with torch.no_grad():
                        _, _, value2_tensor = p1.network(states2)
            

                    Q_tensor, _, _ = p1.network(states) # (batch, 10), (batch, )       

                    valid_Q_tensor = torch.exp(torch.mul(valid_actions, Q_tensor))  
                    Q_action_taken = valid_Q_tensor[torch.arange(total_traj_len), actions]
                    loss =  value_loss_fun(Q_action_taken, rews + p1.gamma * (1-dones) * value2_tensor) 
                    print(loss)
                    loss.backward()
                    optimizer.step()    


                    # Value_step
                    print('Value step')

                    optimizer.zero_grad()
                    with torch.no_grad():
                        Q_tensor, _, _ = p1.network(states) 
                        

                    _, _, value_tensor = p1.network(states) 

                    valid_Q_tensor = torch.exp(torch.mul(valid_actions, Q_tensor)) 
                    valid_policy_tensor = valid_Q_tensor / torch.sum(valid_Q_tensor, dim=1, keepdim=True)

                    target = Q_tensor[torch.arange(total_traj_len), actions] - alpha*torch.log(valid_policy_tensor[torch.arange(total_traj_len), actions])
                    loss = value_loss_fun(target, value_tensor)
                    print(loss)
                    loss.backward()
                    optimizer.step()

            # End epoch
            win_rate = p1.wins/ BATCH_SIZE #total win rate over all games
            print('Win rate: ' + str(win_rate))
            p1.wins = 0
            #p1.empty_buffer()



