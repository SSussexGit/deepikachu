
# coding=utf-8

import time
import sys
import subprocess
import json
import torch
import pprint
import copy
import teams_data

# import custom structures (like MESSAGE, ACTION) and all agents
from custom_structures import *
from agents import *
from state import *
from data_pokemon import *
import neural_net
from neural_net import DeePikachu0

from training import LearningAgent, int_to_action, action_to_int

from game_coordinator import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
VERBOSE = True
MAX_GAME_LEN = 400 #max length is 200 but if you u-turn every turn you move twice per turn

'''
Same as the typical request handling for LearningAgent but doesn do the forward pass, this is done outside
Instead returns state that can be used for neural net
'''
'''
This function then receives the output of the neural net and handles it, returning an action
'''
class ParallelLearningAgent(LearningAgent):

	def __init__(self, id, size, name='Ash', gamma=0.99, lam=0.95):
		# force network=None (can still later store neural net in self.network field)
		super(ParallelLearningAgent, self).__init__(id=id, name=name, size=size, gamma=gamma, lam=lam, network=None)

	def process_request_get_state(self, request):
		self.request_update(request.message)
		message = request.message['request_dict']

		#save the state in the buffer

		#first get our valid action space
		valid_actions = get_valid_actions(self.state, message)

		if (valid_actions == []):
			raise ValueError("no valid actions")
		
		return self.state, valid_actions


	def process_request_receive_tensors(self, valid_actions, policy_tensor, value):
		is_teampreview = ('teamspec' in valid_actions[0])
		if is_teampreview:
			policy_tensor[0:4] *= 0
		else:
			for i in np.arange(10):
				if int_to_action(i) not in valid_actions:
					policy_tensor[i] *= 0

		policy_tensor /= np.sum(policy_tensor)

		#check if we're at teampreview and sample action accordingly. if at teampreview teamspec in first option 
		if is_teampreview:
			action = int_to_action(np.random.choice(np.arange(10), p=policy_tensor), teamprev = True)
		else:
			action = int_to_action(np.random.choice(np.arange(10), p=policy_tensor), teamprev = False)


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
			empty_state[field] = recurse_cat_state(empty_state[field], [state[field] for state in list_of_states])
		else:
			empty_state[field] = np.array([state[field] for state in list_of_states])
	return empty_state


def run_parallel_learning_episode(K, p1s, p2s, network):
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
		sim[k].stdin.write('>start {"formatid":"gen5ou"}\n')
		sim[k].stdin.write('>player p1 {"name":"' + p1s[k].name + '"' + ',"team":"' + teams_data.team1 +'" }\n')
		sim[k].stdin.write('>player p2 {"name":"' + p2s[k].name + '"' + ',"team":"' + teams_data.team1 +'" }\n')
		sim[k].stdin.flush() 

	games = [[] for _ in range(K)]

	p1_outstanding_requests = [[] for _ in range(K)]
	p2_outstanding_requests = [[] for _ in range(K)]

	p1_waiting_for_request_processing = [False for _ in range(K)]
	p2_waiting_for_request_processing = [False for _ in range(K)]

	ended, ended_ctr = [False for _ in range(K)], 0

	if VERBOSE:
		print('[Game threads ended] : ', end='', flush=True)

	# regular game flow
	while True:

		'''
		Idea: simulate K games sequentially until each game has an outstanding request for p1 
		Then forward pass through neural net in batch form 
		(Amortized speed up of factor K in forward pass with GPU)
		'''

		# receive a simulation update and inform players for games not waiting for a request process
		new_messages = [[] for _ in range(K)]
		message_ids =  [set() for _ in range(K)]
		for k in range(K):
			if not p1_waiting_for_request_processing[k] and not p2_waiting_for_request_processing[k] and not ended[k]:
				new = receive_simulator_message(sim[k])
				new_messages[k] += new
				message_ids[k] = retrieve_message_ids_set(sim[k], new)
				games[k] += new
			

		# check if game is over    
		for k in range(K):
			if not p1_waiting_for_request_processing[k] and not p2_waiting_for_request_processing[k] and not ended[k]:
				if 'win' in message_ids[k]:
					# terminate process
					if VERBOSE:
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
					p1s[k].receive_game_update(filter_messages_by_player('p1', new_messages[k]))
					p2s[k].receive_game_update(filter_messages_by_player('p2', new_messages[k]))

					# it is important that this occurs after players receiving the updates (stopped receiving requests)
					# otherwise no more than one request will be read at a time
					if len(p1_outstanding_requests[k]) > 0:
						p1_waiting_for_request_processing[k] = True
					if len(p2_outstanding_requests[k]) > 0:
						p2_waiting_for_request_processing[k] = True

		

		## Player 1 (neural net)
		# process p1 requests as batch for computation speed-up (same network)
		while all([len(p1_outstanding_requests[k]) >= 1 for k in running]):
			
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

			# batch state processing
			with torch.no_grad():
				np_state_cat = recurse_cat_state(create_2D_state(len(running)), np_states)
				policy_tensor, value_tensor = network(np_state_cat)

			# finish up by sampling action
			for idx, k in enumerate(running):
				action = p1s[k].process_request_receive_tensors(valid_actions[idx], policy_tensor[idx].numpy(), value_tensor[idx].numpy())
				send_choice_to_simulator(sim[k], action)

			# if we are about to exit the loop, make sure fully handled games continue to be simulated
			for k in running:
				if len(p1_outstanding_requests[k]) == 0:
					p1_waiting_for_request_processing[k] = False 		
		

		## Player 2 (random agent)
		# simply process all p2 requests per usual 
		# (this can look like for p1 for self-play, though need to be careful with dead-locks where both while loops don't trigger)
		for k in running:
			while p2_outstanding_requests[k]:
				req = p2_outstanding_requests[k].pop(0)
				action = p2s[k].process_request(req)
				send_choice_to_simulator(sim[k], action)

			# p2s[k] ready to continue the simulation
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

	d_player = 32
	d_opp = 16
	d_field = 8

	# init neural net
	p1net = DeePikachu0(state_embedding_settings, d_player=d_player, d_opp=d_opp, d_field=d_field, dropout=0.0)
	p1net = p1net.to(DEVICE)

	EPOCHS = 5
	BATCH_SIZE = 4
	PARELLEL_PER_BATCH = 16
	
	# p1s/p2s are K individual agents storing game information, but the policy/value functions are computed by the same neural net
	p1s = [ParallelLearningAgent(id='p1', name='Red', size = MAX_GAME_LEN*BATCH_SIZE, gamma=0.99, lam=0.95) for _ in range(PARELLEL_PER_BATCH)]
	p2s = [RandomAgent(id='p2', name='Blue') for _ in range(PARELLEL_PER_BATCH)]

	# run games
	for i in range(EPOCHS):

		print(' Epoch {:3d}: '.format(i))
		starttime = time.time()

		p1wins, p2wins = 0, 0

		for j in range(BATCH_SIZE): 

			winner_strings = run_parallel_learning_episode(PARELLEL_PER_BATCH, p1s, p2s, p1net)
			
			for k in range(PARELLEL_PER_BATCH):
				if(winner_strings[k] == p1s[k].name):
					p1wins += 1
				if(winner_strings[k] == p2s[k].name):
					p2wins += 1
				p1s[k].clear_history()
				p2s[k].clear_history()

		endttime = time.time()

		p1winrate = p1wins / (p1wins + p2wins)
		p2winrate = p2wins / (p1wins + p2wins)

		print('[Epoch {:3d}: ave game comp time]  '.format(i) + '{0:.4f}'.format((endttime - starttime)/(BATCH_SIZE * PARELLEL_PER_BATCH)))
		print()
		print('Player 1 | win rate : {0:.4f} |  '.format(p1winrate) + 'wins : {:4d}  '.format(p1wins) + int(50 * p1winrate) * '#')
		print('Player 2 | win rate : {0:.4f} |  '.format(p2winrate) + 'wins : {:4d}  '.format(p2wins) + int(50 * p2winrate) * '#')
		print()