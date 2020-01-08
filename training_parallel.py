
# coding=utf-8
import time
import sys
import subprocess
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pprint
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

from training import LearningAgent, int_to_action, action_to_int, SACAgent, ACTION_SPACE_SIZE

from game_coordinator import *
from game_coordinator_parallel import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_GAME_LEN = 400  # max length is 200 but if you u-turn every turn you move twice per turn
tt_prnt = 10

print(DEVICE)

if __name__ == '__main__':
	'''
	Usage:  ['tester.py', i] for parallel compute
	'''

	if(len(sys.argv)>1):
		c = int(sys.argv[1])
	else:
		c = 0

	torch.manual_seed(c)
	np.random.seed(c)

	state_embedding_settings = {
		'pokemon' :     {'embed_dim' : 3, 'dict_size' : neural_net.MAX_TOK_POKEMON},
		'type' :        {'embed_dim' : 3, 'dict_size' : neural_net.MAX_TOK_TYPE},
		'move' :        {'embed_dim' : 4, 'dict_size' : neural_net.MAX_TOK_MOVE},
		'move_type' :   {'embed_dim' : 3, 'dict_size' : neural_net.MAX_TOK_MOVE_TYPE},
		'ability' :     {'embed_dim' : 2, 'dict_size' : neural_net.MAX_TOK_ABILITY},
		'item' :        {'embed_dim' : 2, 'dict_size' : neural_net.MAX_TOK_ITEM},
		'condition' :   {'embed_dim' : 2, 'dict_size' : neural_net.MAX_TOK_CONDITION},
		'weather' :     {'embed_dim' : 2, 'dict_size' : neural_net.MAX_TOK_WEATHER},
		'alive' :       {'embed_dim' : 1, 'dict_size' : neural_net.MAX_TOK_ALIVE},
		'disabled' :    {'embed_dim' : 1, 'dict_size' : neural_net.MAX_TOK_DISABLED},
		'spikes' :      {'embed_dim' : 1, 'dict_size' : neural_net.MAX_TOK_SPIKES},
		'toxicspikes' : {'embed_dim' : 1, 'dict_size' : neural_net.MAX_TOK_TOXSPIKES},
		'fieldeffect' : {'embed_dim' : 1, 'dict_size' : neural_net.MAX_TOK_FIELD},
	}	



	EPOCHS = 50
	BATCH_SIZE = 16
	PARELLEL_PER_BATCH = 64
	gamma = 0.99
	lam = 0.95
	verbose = True

	alpha = 0.05
	warmup_epochs = 5 # number of epochs playing randomly
	train_update_iters = 100

	# neural nets
	d_player = 16
	d_opp = 16
	d_field = 4

	p1net = DeePikachu0(
		state_embedding_settings,
		d_player=d_player,
		d_opp=d_opp,
		d_field=d_field,
		dropout=0.0,
		attention=True)
	p1net = p1net.to(DEVICE)

	v_target_net = copy.deepcopy(p1net)
	v_target_net.to(DEVICE)

	# experience replay
	replay_size = 1e6
	minibatch_size = 1000  # number of examples sampled from experience replay in each update
	replay = ExperienceReplay(size=int(replay_size), minibatch_size=minibatch_size)

	# agents
	p1s = [ParallelLearningAgent(
		id='p1', name='Red', size=MAX_GAME_LEN + 1, gamma=gamma, lam=lam, alpha=alpha) for _ in range(PARELLEL_PER_BATCH)]
	p2s = [RandomAgent(id='p2', name='Blue') for _ in range(PARELLEL_PER_BATCH)]

	# game mode
	formatid = 'gen5ou'  # 'gen5ou' 'gen5randombattle'

	player_teams = teams_data.team1 #None #teams_data.team1
	

	# optimizer 
	lr = 0.0001 #previously used 0.001, 0.0004 (SAC paper recommendations)
	weight_decay = 1e-4
	optimizer = optim.Adam(p1net.parameters(), lr=lr, weight_decay=weight_decay)


	mse_loss = nn.MSELoss(reduction='mean')

	# training loop
	max_winrate = 0
	train_win_array = []
	eval_win_array = []

	print(f'\nEpochs: {EPOCHS}\nGames per epoch: {BATCH_SIZE * PARELLEL_PER_BATCH}\n'
		  f'(batch size: {BATCH_SIZE}; in parallel: {PARELLEL_PER_BATCH})\n')

	# simulate `EPOCHS` epochs
	for i in range(EPOCHS):
		p1net.train()
		print(' Epoch {:3d}: '.format(i))

		p1wins, p2wins = 0, 0

		# warmup mode
		if(i >= warmup_epochs):
			for k in range(PARELLEL_PER_BATCH):
				p1s[k].warmup = False
		else:
			for k in range(PARELLEL_PER_BATCH):
				p1s[k].warmup = True

		# simulate `BATCH_SIZE` * `PARELLEL_PER_BATCH` games parallelized and store result in replay
		for j in range(BATCH_SIZE):
			winner_strings = run_parallel_learning_episode(
				PARELLEL_PER_BATCH, p1s, p2s, p1net, formatid=formatid, team=player_teams, verbose=verbose)
			
			for k in range(PARELLEL_PER_BATCH):
				if(winner_strings[k] == p1s[k].name):
					p1wins += 1
				if(winner_strings[k] == p2s[k].name):
					p2wins += 1

				p1s[k].clear_history()
				p2s[k].clear_history()
				p1s[k].end_traj()

				# for player in every game, empty the buffers into experience replay
				states, states2, actions, advs, rtgs, logps, valid_actions, rews, dones = p1s[k].spit()
				replay.swallow(states, states2, actions, advs, rtgs, logps, valid_actions, rews, dones)
				
				p1s[k].empty_buffer()

		# perform updates on neural net
		if(replay.total_tuples > replay.minibatch_size):
			
			for tt in range(train_update_iters):

				'''
				Soft Actor critic 
				(Discrete, so no policy net)
				'''

				# Random sample from buffer (experience replay)
				states, states2, actions, advs, rtgs, logps, valid_actions, rews, dones = replay.get()
				actions = torch.tensor(actions, dtype=torch.long)
				advs = torch.tensor(advs, dtype=torch.float)
				rtgs = torch.tensor(rtgs, dtype=torch.float)
				logps = torch.tensor(logps, dtype=torch.float)
				valid_actions = torch.tensor(valid_actions, dtype=torch.float).to(DEVICE)
				rews = torch.tensor(rews, dtype=torch.float).to(DEVICE)
				dones = torch.tensor(dones, dtype=torch.float).to(DEVICE)

				total_traj_len = actions.shape[0]

				# compute supervised learning targets
				with torch.no_grad():
					
					# value function target for s'
					v_target_net.eval()
					_, _, v_tensor_fixed = v_target_net(states2)
					v_target_net.train()

					# q function for s, a pairs
					p1net.eval()
					q_tensor_A_fixed, q_tensor_B_fixed, _ = p1net(states)
					p1net.train()


					# q function regression target
					q_target = rews + p1s[0].gamma * (1 - dones) * v_tensor_fixed
					
					# v function regression target (min over both q heads:)
					# 1
					valid_q_A = torch.mul(valid_actions, torch.exp((q_tensor_A_fixed-torch.mean(q_tensor_A_fixed, dim=1, keepdim=True)) / alpha))
					valid_policy_A = valid_q_A / valid_q_A.sum(dim=1, keepdim=True)

					actions_tilde = torch.distributions.Categorical(probs=valid_policy_A).sample()

					v_target_A = q_tensor_A_fixed[torch.arange(total_traj_len), actions_tilde] \
						- alpha * torch.log(valid_policy_A[torch.arange(total_traj_len), actions_tilde])

					# 2
					v_target_B = q_tensor_B_fixed[torch.arange(total_traj_len), actions_tilde] \
						- alpha * torch.log(valid_policy_A[torch.arange(total_traj_len), actions_tilde])
					
					# min
					v_target = torch.min(torch.stack([v_target_A, v_target_B], dim=1), dim=1)[0]

				# run updates on the networks
				p1net.train()

				# Q step A
				optimizer.zero_grad()
				q_tensor_A, _, _ = p1net(states)
				q_action_taken_A = q_tensor_A[torch.arange(total_traj_len), actions]

				loss = mse_loss(q_action_taken_A, q_target)
				loss.backward()
				optimizer.step()  

				if (tt % tt_prnt == 0):
					print('Q step A: ', loss.detach().item(), end='\t')

				# Q step B
				optimizer.zero_grad()
				_, q_tensor_B, _ = p1net(states)
				q_action_taken_B = q_tensor_B[torch.arange(
					total_traj_len), actions]

				loss = mse_loss(q_action_taken_B, q_target)
				loss.backward()
				optimizer.step()

				if (tt % tt_prnt == 0):
					print('Q step B: ', loss.detach().item(), end='\t')

				# V step
				optimizer.zero_grad()
				_, _, value_tensor = p1net(states)
				
				loss = mse_loss(value_tensor, v_target)
				loss.backward()
				optimizer.step()

				if (tt % tt_prnt == 0):
					print('V step: ', loss.detach().item(), end='\n')

				# Update target network for value function using exponential moving average
				
				with torch.no_grad():
							
					polyak = 0.995 # (default in openai pseudocode)
					for param, param_target in zip(p1net.parameters(), v_target_net.parameters()):
						param_target.data.copy_(polyak * param_target.data + (1 - polyak) * param.data)

		# End epoch
		train_win_rate = float(p1wins) / float(p1wins + p2wins) 
		print('Train win rate (batch): ' + str(train_win_rate))
		for k in range(PARELLEL_PER_BATCH):
			p1s[k].wins = 0
		train_win_array.append(train_win_rate)

		# do an eval epoch
		if (i % 5 == 4):

			# agent plays argmax of q function
			p1net.eval()
			for k in range(PARELLEL_PER_BATCH):
				p1s[k].evalmode = True
				p1s[k].warmup = False
				p1s[k].wins = 0

			p1wins_eval, p2wins_eval = 0, 0

			for j in range(BATCH_SIZE): 
				winner_strings = run_parallel_learning_episode(PARELLEL_PER_BATCH, p1s, p2s, p1net)
				
				for k in range(PARELLEL_PER_BATCH):
					if(winner_strings[k] == p1s[k].name):
						p1wins_eval += 1
					if(winner_strings[k] == p2s[k].name):
						p2wins_eval += 1

					p1s[k].end_traj()
					p1s[k].clear_history()
					p2s[k].clear_history()

					# empty the player buffers without storing in replay
					p1s[k].empty_buffer()

			p1net.train()
			for k in range(PARELLEL_PER_BATCH):
				p1s[k].evalmode = False

			p1winrate_eval = float(p1wins_eval) / float(p1wins_eval + p2wins_eval)
			p2winrate_eval = float(p2wins_eval) / float(p1wins_eval + p2wins_eval)
		
			max_eval_winrate = max(p1winrate_eval, max_winrate)

			print('\n[Epoch {:3d}: Evaluation]  \n'.format(i) )
			print('Player 1 | win rate : {0:.4f} |  '.format(p1winrate_eval) + 'wins : {:4d}  '.format(p1wins_eval) + int(50 * p1winrate_eval) * '#')
			print('Player 2 | win rate : {0:.4f} |  '.format(p2winrate_eval) + 'wins : {:4d}  '.format(p2wins_eval) + int(50 * p2winrate_eval) * '#')
			print()

			if(p1winrate_eval >= max_eval_winrate):
				torch.save(p1net.state_dict(), 'output/network_'+ str(c)+'_'+str(i)+'.pth')
			eval_win_array.append(p1winrate_eval)

	with open('output/eval_results' + str(c) + '.csv', 'w') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(eval_win_array)

	with open('output/train_results' + str(c) + '.csv', 'w') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(train_win_array)



