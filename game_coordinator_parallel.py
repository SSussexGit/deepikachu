
# coding=utf-8
import time
import sys
import subprocess
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
import copy
import teams_data
import csv

# import custom structures (like MESSAGE, ACTION) and all agents
from custom_structures import *
from agents import *
from state import *
from data_pokemon import *
import neural_net
from neural_net import DeePikachu0
import teams_data

from training import LearningAgent, int_to_action, action_to_int, SACAgent, ACTION_SPACE_SIZE

from game_coordinator import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_GAME_LEN = 400  # max length is 200 but if you u-turn every turn you move twice per turn
MAX_GAME_CUTOFF = 395 # 395  # cutoff to avoid `Battle.maybeTriggerEndlessBattleClause` error

class ExperienceReplay:
	def __init__(self, size=20000, minibatch_size=100, gamma=0.99):

		self.replay_size = size
		self.state_replay = create_2D_state(self.replay_size)
		self.state2_replay = create_2D_state(self.replay_size)
		# won't overflow since max number is >> possible length
		self.action_replay = np.zeros(self.replay_size, dtype=int)
		self.adv_replay = np.zeros(self.replay_size, dtype=np.float32)
		self.rew_replay = np.zeros(self.replay_size, dtype=np.float32)
		self.rtg_replay = np.zeros(
			self.replay_size, dtype=np.float32)  # the rewards-to-go
		# save in np because we recompute value a bunch anyway
		self.val_replay = np.zeros(self.replay_size, dtype=np.float32)
		self.logp_replay = np.zeros(self.replay_size, dtype=np.float32)  # logp value
		# stores what actions were valid at that time point as a 1hot
		self.valid_actions_replay = np.zeros((self.replay_size, ACTION_SPACE_SIZE))
		self.gamma = gamma
		self.total_tuples = 0  # so we know where to cut off vectors above for updates
		self.ptr_start = 0  # an index of the start of the trajectory currently being put in memory
		self.ptr = 0  # an index of the next tuple to be put in the buffer
		self.done_replay = np.zeros(self.replay_size, dtype=int)

		self.minibatch_size = minibatch_size

	def store_in_replay(self, state, state2, action, logp, valid_actions, rew, done):
		'''
		Stores everything in the replay and increments the pointer
		'''

		self.state_replay = self.recurse_store_state(
			self.state_replay, state, self.ptr)

		self.state2_replay = self.recurse_store_state(
			self.state2_replay, state2, self.ptr)

		self.action_replay[self.ptr] = action

		self.logp_replay[self.ptr] = logp

		self.rew_replay[self.ptr] = rew

		self.done_replay[self.ptr] = done

		#self.valid_actions[self.ptr]
		for action_index in range(ACTION_SPACE_SIZE):
			if(valid_actions[action_index] == 1):
				self.valid_actions_replay[self.ptr, action_index] = 1
			else:
				self.valid_actions_replay[self.ptr, action_index] = 0

		self.ptr += 1
		if(self.ptr == self.replay_size):
			self.ptr = 0
		# if full stay full else increment with pointer
		self.total_tuples = max(self.ptr, self.total_tuples)
		return

	def recurse_store_state(self, state_buffer, state, index):
		'''
		stores a state in buffer recursively
		'''
		for field in state:
			if (isinstance(state[field], dict)):
				state_buffer[field] = self.recurse_store_state(
					state_buffer[field], state[field], index)
			else:
				state_buffer[field][index] = state[field]
		return state_buffer

	def recurse_unfold_state(self, state_holder, states, index):
		'''
        extracts a state at a specific index from a buffer
        '''
		for field in state_holder:
		    if (isinstance(state_holder[field], dict)):
		        state_holder[field] = self.recurse_unfold_state(
		        	state_holder[field], states[field], index)
		    else:
		        state_holder[field] = states[field][index]
		return state_holder

	def swallow(self, states, states2, actions, advs, rtgs, logps, valid_actions, rews, dones):
		"""
		Stores buffer info in the replay
		"""
		self.ptr_start = self.ptr
		for i in range(actions.shape[0]):
			state_i = self.recurse_unfold_state(copy.deepcopy(default_state), states, i)
			state2_i = self.recurse_unfold_state(
				copy.deepcopy(default_state), states2, i)
			self.store_in_replay(
				state_i, state2_i, actions[i], logps[i], valid_actions[i], rews[i], dones[i])

	def recurse_index_state(self, state_replay, idxs):
	    '''
	    stores a state in replay recursively
	    '''
	    for field in state_replay:
	        if (isinstance(state_replay[field], dict)):
	            state_replay[field] = self.recurse_index_state(
	            	state_replay[field], idxs)
	        else:
	            state_replay[field] = state_replay[field][idxs]
	    return state_replay

	def get(self):
		'''
		Returns a sample from the experience replay
		'''
		idxs = np.random.randint(0, self.total_tuples, size=self.minibatch_size)
		return [self.recurse_index_state(copy.deepcopy(self.state_replay), idxs), self.recurse_index_state(copy.deepcopy(self.state2_replay), idxs), self.action_replay[idxs], self.adv_replay[idxs],
                    self.rtg_replay[idxs], self.logp_replay[idxs], self.valid_actions_replay[idxs], self.rew_replay[idxs], self.done_replay[idxs]]


class ParallelLearningAgent(SACAgent):

	def __init__(self, id, size, name='Ash', gamma=0.99, lam=0.95, alpha=0.05):
		# force network=None (can still later store neural net in self.network field)
		super(ParallelLearningAgent, self).__init__(
			id=id, name=name, size=size, gamma=gamma, lam=lam, network=None, alpha=alpha)

	def process_request_get_state(self, request):
		'''
		Same as the typical request handling for LearningAgent but doesn do the forward pass, this is done outside
		Instead returns state that can be used for neural net
		'''
		self.request_update(request.message)
		message = request.message['request_dict']
		#print(message)
		#save the state in the buffer

		#first get our valid action space
		valid_actions = get_valid_actions(self.state, message)

		if (valid_actions == []):
			raise ValueError("no valid actions")

		return self.state, valid_actions

	def process_request_receive_tensors(self, valid_actions, q_tensor, value):
		'''
		This function then receives the output of the neural net and handles it, returning an action
		'''
		'''
		print(self.state['player']['active'])
		for i in range(6):
			print(self.state['player']['team'][i])
		print()
		print(self.state['opponent']['active'])
		for i in range(6):
			print(self.state['opponent']['team'][i])
		print()
		print()
		'''
		#print(q_tensor)
		#print(value)
		is_teampreview = ('teamspec' in valid_actions[0])
		if(self.warmup):
			action = random.choice(valid_actions)
			logp = np.log(1/min(1, len(valid_actions)))

		else:
			valid_actions_np = np.ones((10))
			if is_teampreview:
				for i in np.arange(10):
					if int_to_action(i, teamprev=True) not in valid_actions:
						q_tensor[i] = -np.inf
						valid_actions_np[i] = 0
			else:
				for i in np.arange(10):
					if int_to_action(i) not in valid_actions:
						q_tensor[i] = -np.inf
						valid_actions_np[i] = 0


			if((not self.evalmode) and (np.random.binomial(1, self.alpha) == 1)):
				action_to_choose = np.random.choice(10, p=valid_actions_np/np.sum(valid_actions_np))
			else:
				action_to_choose = np.argmax(q_tensor)

			
			action = int_to_action(action_to_choose, teamprev=is_teampreview)

			#save logpaction in buffer (not really needed since it gets recomputed)
			logp = np.log(1/min(1, len(valid_actions)))

		self.store_in_buffer(self.state, action, value, logp, valid_actions)

		return PlayerAction(self.id, action)


def recurse_cat_state(empty_state, list_of_states):
	'''
	stores a state in buffer recursively
	'''
	for field in empty_state:
		if (isinstance(empty_state[field], dict)):
			empty_state[field] = recurse_cat_state(
				empty_state[field], [state[field] for state in list_of_states])
		else:
			empty_state[field] = np.array([state[field] for state in list_of_states])
	return empty_state


def run_parallel_learning_episode(K, p1s, p2s, network, formatid, player_team_size, verbose=True):
	'''
	takes in 2 agents and plays K games between them in parallel (one forward pass of network)
	Assumes p1s are ParallelLearningAgent
	'''
	for k in range(K):
		assert(isinstance(p1s[k], ParallelLearningAgent))

	# opens: `./pokemon-showdown simulate-battle` K times
	sim = [
		subprocess.Popen('./pokemon-showdown simulate-battle',
                   shell=True,
                   stdin=subprocess.PIPE,
                   stdout=subprocess.PIPE,
                   universal_newlines=True) for _ in range(K)
	]

	# start all games
	for k in range(K):
		# to add random seed do "seed":"[1234,5678,9012,3456]"
		sim[k].stdin.write('>start {"formatid":"' + (formatid) + '"}\n')

		if player_team_size:

			# sample random team of size player_team_size
			team1 = teams_data.get_random_team(player_team_size)
			team2 = teams_data.get_random_team(player_team_size)

			# team1 = teams_data.team3
			# team2 = teams_data.team3
			# print('Starting game with team: ')
			# print(team1)
			# print(team2)

			sim[k].stdin.write(
				'>player p1 {"name":"' + p1s[k].name + '"' + ',"team":"' + (team1) + '" }\n')
			sim[k].stdin.write(
				'>player p2 {"name":"' + p2s[k].name + '"' + ',"team":"' + (team2) + '" }\n')

		else:
			sim[k].stdin.write('>player p1 {"name":"' + p1s[k].name + '" }\n')
			sim[k].stdin.write('>player p2 {"name":"' + p2s[k].name + '" }\n')

		sim[k].stdin.flush()

	# game messages
	games = [[] for _ in range(K)]

	# outstanding requests for players
	p1_outstanding_requests = [[] for _ in range(K)]
	p2_outstanding_requests = [[] for _ in range(K)]

	# true if and only if 1) at least 1 outstanding request for player
	# AND 2) corresponding game updates have been sent to player (for temporal ordering)
	p1_waiting_for_request_processing = [False for _ in range(K)]
	p2_waiting_for_request_processing = [False for _ in range(K)]

	ended, ended_ctr, turn_ctr = [False for _ in range(K)], 0, [0 for _ in range(K)]

	if verbose:
		print('[Game threads ended] : ', end='', flush=True)

	# regular game flow
	while True:
		'''
		Idea: simulate K games sequentially until each game has an outstanding request for p1 
		Then forward pass through neural net in batch form 
		(Amortized speed up of factor K in forward pass with GPU)
		'''

		# check if max turn count is reached and if so restart game (to avoid endlessBattle error)
		for k in range(K):
			if turn_ctr[k] > MAX_GAME_CUTOFF:

				# log
				print('EARLYCUTOFF: Game was cut off early.')
				with open(f'output/cutoff_log_k={k}.txt', 'w') as f:
					for m in games[k]:
						f.write(m.original_str + '\n')

				# restart game
				assert(not ended[k])

				turn_ctr[k] = 0
				games[k] = []
				p1_outstanding_requests[k] = []
				p2_outstanding_requests[k] = []
				p1_waiting_for_request_processing[k] = []
				p2_waiting_for_request_processing[k] = []

				sim[k].terminate()
				sim[k].stdin.close()
				sim[k] = None
				sim[k] = subprocess.Popen('./pokemon-showdown simulate-battle',
                   shell=True,
                   stdin=subprocess.PIPE,
                   stdout=subprocess.PIPE,
                   universal_newlines=True)
				sim[k].stdin.write('>start {"formatid":"' + (formatid) + '"}\n')
				if player_team_size:
					team1 = teams_data.get_random_team(player_team_size)
					team2 = teams_data.get_random_team(player_team_size)
					sim[k].stdin.write(
						'>player p1 {"name":"' + p1s[k].name + '"' + ',"team":"' + (team1) + '" }\n')
					sim[k].stdin.write(
						'>player p2 {"name":"' + p2s[k].name + '"' + ',"team":"' + (team2) + '" }\n')
				else:
					sim[k].stdin.write('>player p1 {"name":"' + p1s[k].name + '" }\n')
					sim[k].stdin.write('>player p2 {"name":"' + p2s[k].name + '" }\n')
				sim[k].stdin.flush()

				# restart agent
				p1s[k].clear_history()
				p1s[k].end_traj()
				p1s[k].empty_buffer()
				p2s[k].clear_history()


		# receive a simulation update and inform players for games not waiting for a request process
		new_messages = [[] for _ in range(K)]
		message_ids = [set() for _ in range(K)]
		for k in range(K):
			if not p1_waiting_for_request_processing[k] and not p2_waiting_for_request_processing[k] and not ended[k]:
				new = receive_simulator_message(sim[k])
				message_ids[k] = retrieve_message_ids_set(sim[k], new)
				for m in new:
					new_messages[k].append(m)
					games[k].append(m)

					# update turn count
					if m.message['id'] == 'turn':
						turn_ctr[k] = int(m.message['number'])

				# # DEBUG
				# for m in new:
				# 	if not m.message['id'] == 'request':
				# 		print(m.original_str)
				# 	else:
				# 		print('|request| ', m.message['request_dict'].keys())

		# check if game is over
		for k in range(K):
			if not p1_waiting_for_request_processing[k] and not p2_waiting_for_request_processing[k] and not ended[k]:
				if 'win' in message_ids[k]:
					# terminate process
					if verbose:
						if ended_ctr == 0:
							print(f'{k}', end='', flush=True)
						elif ended_ctr == K - 1:
							print(f', {k}', end='\n', flush=True)
						else:
							print(f', {k}', end='', flush=True)

					ended[k] = True
					ended_ctr += 1
					sim[k].terminate()
					sim[k].stdin.close()

		# check what games are still running
		if ended_ctr == K:
			break
		else:
			running = []
			for k in range(K):
				if not ended[k]:
					running.append(k)

		# process new messages
		for k in running:
			if not p1_waiting_for_request_processing[k] and not p2_waiting_for_request_processing[k]:
				# if there are requests, record them for corresponding player (if not `wait` request)
				if 'request' in message_ids[k]:
					new_requests = filter_messages_by_id('request', new_messages[k])
					for new in new_requests:
						if not 'wait' in new.message['request_dict'].keys():
							pl = new.adressed_players
							if len(pl) != 1:
								raise ValueError('Requests should be addressed to exactly one player')
							pl = pl[0]
							if pl == 'p1':
								p1_outstanding_requests[k].append(new)
							else:
								p2_outstanding_requests[k].append(new)

				# regular message: send updates to players
				else:

					# 1) update players on new information
					p1s[k].receive_game_update(
						filter_messages_by_player('p1', new_messages[k]))
					p2s[k].receive_game_update(
						filter_messages_by_player('p2', new_messages[k]))

					# it is important that this occurs after players receiving the updates (i.e. stopped receiving requests)
					# otherwise a) no more than one request will be read at a time, b) request will be sent before game updates
					if len(p1_outstanding_requests[k]) > 0:
						p1_waiting_for_request_processing[k] = True
					if len(p2_outstanding_requests[k]) > 0:
						p2_waiting_for_request_processing[k] = True

		## Player 1 (neural net)
		# process p1 requests as batch for computation speed-up (same network)
		while all([p1_waiting_for_request_processing[k] for k in running]):

			# at this point all requests should be for p1s and not contain `wait`
			# get recent requests for each thread
			reqs = [p1_outstanding_requests[k].pop(0) for k in running]

			# request one forward pass in batch from neural network
			np_states, valid_actions = [], []
			for idx, k in enumerate(running):
				st, va = p1s[k].process_request_get_state(reqs[idx])
				# print(json.dumps(st, indent=2))
				np_states.append(st)
				valid_actions.append(va)

			# DEBUG
			# print('---- BEFORE NN')
			# for j in range(4):
			# 	pprint.pprint(np_states[0]['player']['active']['moves'][j])

			# batch state processing
			with torch.no_grad():
				np_state_cat = recurse_cat_state(create_2D_state(len(running)), np_states)
				q_tensor, _, value_tensor = network(np_state_cat)

			# finish up by sampling action
			for idx, k in enumerate(running):
				action = p1s[k].process_request_receive_tensors(
					valid_actions[idx], q_tensor[idx].cpu().numpy(), value_tensor[idx].cpu().numpy())
				send_choice_to_simulator(sim[k], action)

			# if we are about to exit the loop, make sure fully handled games continue to be simulated
			for k in running:
				if len(p1_outstanding_requests[k]) == 0:
					p1_waiting_for_request_processing[k] = False

		## Player 2 (random agent)
		# simply process all p2 requests per usual
		# (this can look like for p1 for self-play, though need to be careful with dead-locks
		# where both while loops don't trigger because waiting for requests on 2 different threads, if that can happen)
		for k in running:
			while p2_waiting_for_request_processing[k]:
				req = p2_outstanding_requests[k].pop(0)
				action = p2s[k].process_request(req)
				send_choice_to_simulator(sim[k], action)

				# p2s[k] ready to continue the simulation
				if len(p2_outstanding_requests[k]) == 0:
					p2_waiting_for_request_processing[k] = False

	# collect results
	winner_strings = []
	for k in range(K):
		game_over_message = filter_messages_by_id('win', games[k])[0]
		winner_string = game_over_message.message['info_json']['winner']
		if(winner_string == p1s[k].name):
			p1s[k].won_game()
		elif (winner_string == p2s[k].name):
			p2s[k].won_game()
		else:
			raise ValueError('Unknown winner.')
		winner_strings.append(winner_string)

	return winner_strings


if __name__ == '__main__':
	'''
	Usage:  ['tester.py', i] for parallel compute
	'''

	if(len(sys.argv) > 1):
		c = int(sys.argv[1])
	else:
		c = 0

	torch.manual_seed(c)
	np.random.seed(c)

	state_embedding_settings = {
		'pokemon':     {'embed_dim': 32, 'dict_size': neural_net.MAX_TOK_POKEMON},
		'type':        {'embed_dim': 8, 'dict_size': neural_net.MAX_TOK_TYPE},
		'move':        {'embed_dim': 8, 'dict_size': neural_net.MAX_TOK_MOVE},
		'move_type':   {'embed_dim': 8, 'dict_size': neural_net.MAX_TOK_MOVE_TYPE},
		'ability':     {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_ABILITY},
		'item':        {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_ITEM},
		'condition':   {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_CONDITION},
		'weather':     {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_WEATHER},
		'alive':       {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_ALIVE},
		'disabled':    {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_DISABLED},
		'spikes':      {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_SPIKES},
		'toxicspikes': {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_TOXSPIKES},
		'fieldeffect': {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_FIELD},
	}

	d_player = 16
	d_opp = 16
	d_field = 8

	# init neural net
	p1net = DeePikachu0(
		state_embedding_settings,
		d_player=d_player,
		d_opp=d_opp,
		d_field=d_field,
		dropout=0.0,
		attention=True)
	p1net = p1net.to(DEVICE)

	BATCH_SIZE = 3
	PARELLEL_PER_BATCH = 8

	# p1s/p2s are K individual agents storing game information, but the policy/value functions are computed by the same neural net
	p1s = [ParallelLearningAgent(id='p1', name='Red', size=20000,
	                             gamma=0.99, lam=0.95) for _ in range(PARELLEL_PER_BATCH)]
	p2s = [RandomAgent(id='p2', name='Blue') for _ in range(PARELLEL_PER_BATCH)]

	for j in range(BATCH_SIZE):

		winner_strings = run_parallel_learning_episode(
			PARELLEL_PER_BATCH, p1s, p2s, p1net)

		for k in range(PARELLEL_PER_BATCH):

			p1s[k].clear_history()
			p2s[k].clear_history()
			p1s[k].end_traj()
			p1s[k].empty_buffer()

		print('Winner strings:', winner_strings)
