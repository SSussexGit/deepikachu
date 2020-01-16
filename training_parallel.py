
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

import neural_net_small_2
from neural_net_small_2 import SmallDeePikachu2

from training import LearningAgent, int_to_action, action_to_int, SACAgent, ACTION_SPACE_SIZE

from game_coordinator import *
from game_coordinator_parallel import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_GAME_LEN = 400  # max length is 200 but if you u-turn every turn you move twice per turn
SAVE_ROOT = 'output/'

print(f'device = {DEVICE}', flush=True)

def save_model(fname, c, i, model, model_target, optimizer):
    torch.save({
		'seed' : c,
		'epoch' : i,
		'torch_rng_state': torch.get_rng_state(),
		'numpy_rng_state': np.random.get_state(),
		'model_state_dict': model.state_dict(),
		'target_state_dict': model_target.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
    }, SAVE_ROOT + fname + '_c=' + str(c) + '_ep=' + str(i) + '.pt')

def load_model(fname, model, model_target, optimizer):
	checkpoint = torch.load(SAVE_ROOT + fname + '.pt', map_location = torch.device(DEVICE))
	c = checkpoint['seed']
	i = checkpoint['epoch']
	torch_rng_state = checkpoint['torch_rng_state']
	numpy_rng_state = checkpoint['numpy_rng_state']
	torch.set_rng_state(torch_rng_state)
	np.random.set_state(numpy_rng_state)
	model.load_state_dict(checkpoint['model_state_dict'])
	model_target.load_state_dict(checkpoint['target_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return model, model_target, optimizer, c, i


def train_parallel_epochs(p1s, p2s, optimizer, p1net, v_target_net, replay,
	formatid, player_team_size,
	alpha, warmup_epochs, train_update_iters,
	epochs, batch_size, parallel_per_batch, 
	starting_epoch, eval_epoch_every,
	gamma, lam,
	fstring, tt_print=10, verbose=True):

	# info
	# print(p1net)
	print(p1net.state_embedding_settings)
	print(p1net.hidden_layer_settings)
	print()
	print(f'activation = {p1net.f_activation}')
	print(f'move module is identity = {p1net.move_identity}')
	print(f'layer norm = {p1net.layer_norm}')
	print(f'gamma = {gamma}')
	print(f'lam = {lam}')
	print(f'alpha = {alpha}')
	print(f'formatid = {formatid}')
	print(f'player_team_size = {player_team_size}')
	print(f'warmup_epochs = {warmup_epochs}')
	print(f'train_update_iters = {train_update_iters}')
	print(f'lr = {optimizer.param_groups[0]["lr"]}')
	print(f'weight_decay = {optimizer.param_groups[0]["weight_decay"]}')
	print(f'replay_size = {replay.replay_size}')
	print(f'replay_minibatch = {replay.minibatch_size}')

	mse_loss = nn.MSELoss(reduction='mean')

	# training loop
	max_winrate = 0
	train_win_array = []
	eval_win_array = []

	print()
	print(f'epochs = {epochs}')
	print(f'initialized at epoch = {starting_epoch}')
	print(f'games per epoch = {batch_size * parallel_per_batch} (batch: {batch_size}, parallel: {parallel_per_batch})')
	print(flush=True)


	# simulate `EPOCHS` epochs
	for i in range(starting_epoch, epochs + starting_epoch):
		p1net.train()

		p1wins, p2wins = 0, 0

		# warmup mode
		if(i >= warmup_epochs):
			for k in range(parallel_per_batch):
				p1s[k].warmup = False
				p1s[k].alpha = max(0.87*p1s[k].alpha, 0.05)
		else:
			for k in range(parallel_per_batch):
				p1s[k].warmup = True

		# simulate `BATCH_SIZE` * `PARELLEL_PER_BATCH` games parallelized and store result in replay
		for j in range(batch_size):
			winner_strings = run_parallel_learning_episode(
				parallel_per_batch, p1s, p2s, p1net, formatid=formatid, player_team_size=player_team_size, verbose=False, train=True)

			for k in range(parallel_per_batch):
				if(winner_strings[k] == p1s[k].name):
					p1wins += 1
				if(winner_strings[k] == p2s[k].name):
					p2wins += 1

				p1s[k].clear_history()
				p2s[k].clear_history()
				p1s[k].end_traj()

				# for player in every game, empty the buffers into experience replay
				states, states2, actions, advs, rtgs, logps, valid_actions, rews, dones = p1s[k].spit()
				replay.swallow(states, states2, actions, advs, rtgs,
				               logps, valid_actions, rews, dones)

				p1s[k].empty_buffer()

		# End epoch
		train_win_rate = float(p1wins) / float(p1wins + p2wins)
		print('Epoch {:3d}:  train win rate: {}'.format(i, train_win_rate) + ('  (warm-up)' if (i < warmup_epochs) else ''), flush=True)

		for k in range(parallel_per_batch):
			p1s[k].wins = 0

		train_win_array.append(train_win_rate)

		# perform updates on neural net
		if(replay.total_tuples > replay.minibatch_size):

			q_losses_A, q_losses_B, v_losses = [], [], []

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
				rews = 10 * torch.tensor(rews, dtype=torch.float).to(DEVICE)
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

					assert(torch.isnan(v_tensor_fixed).sum() == 0)
					assert(torch.isnan(q_tensor_A_fixed).sum() == 0)
					assert(torch.isnan(q_tensor_B_fixed).sum() == 0)

					# q function regression target
					q_target = rews + p1s[0].gamma * (1 - dones) * v_tensor_fixed

					# v function regression target (min over both q heads:)
					# 1

					masked_q_A = torch.mul(valid_actions, q_tensor_A_fixed)

					v_target_A = torch.max(masked_q_A, dim=1)[0]

					# 2
					masked_q_B = torch.mul(valid_actions, q_tensor_B_fixed)

					v_target_B = torch.max(masked_q_B, dim=1)[0]
					# min
					v_target = torch.min(torch.stack(
						[v_target_A, v_target_B], dim=1), dim=1)[0]

				# run updates on the networks
				p1net.train()

				# Q step A
				optimizer.zero_grad()
				q_tensor_A, _, _ = p1net(states)
				q_action_taken_A = q_tensor_A[torch.arange(total_traj_len), actions]

				loss = mse_loss(q_action_taken_A, q_target)
				loss.backward()
				optimizer.step()
				
				q_losses_A.append(loss.detach().item())
				if (tt % tt_print == tt_print - 1 and verbose):
					print('{:3d}'.format(tt), end='\t')
					print('Q step A: {:.8f}'.format(sum(q_losses_A) / len(q_losses_A)), end='\t')
					q_losses_A = []
				

				# Q step B
				optimizer.zero_grad()
				_, q_tensor_B, _ = p1net(states)
				q_action_taken_B = q_tensor_B[torch.arange(
					total_traj_len), actions]

				loss = mse_loss(q_action_taken_B, q_target)
				loss.backward()
				optimizer.step()
				
				q_losses_B.append(loss.detach().item())
				if (tt % tt_print == tt_print - 1 and verbose):
					print('Q step B: {:.8f}'.format(sum(q_losses_B) / len(q_losses_B)), end='\t')
					q_losses_B = []
				
				# V step
				optimizer.zero_grad()
				_, _, value_tensor = p1net(states)

				loss = mse_loss(value_tensor, v_target)
				loss.backward()
				optimizer.step()

				v_losses.append(loss.detach().item())
				if (tt % tt_print ==  tt_print - 1 and verbose):
					print('V step: {:.8f}'.format(sum(v_losses) / len(v_losses)), end='\n', flush=True)
					v_losses = []

				# Update target network for value function using exponential moving average
				with torch.no_grad():

					polyak = 0.995  # (default in openai pseudocode)
					for param, param_target in zip(p1net.parameters(), v_target_net.parameters()):
						param_target.data.copy_(
							polyak * param_target.data + (1 - polyak) * param.data)

		# DEBUG
		print('Debug printout')
		with torch.no_grad():
			prn = 3
			tmpqa, tmpqb, tmpv = p1net(states)
			if isinstance(p1net, SmallDeePikachu2):
				print('Player pkmn alive status: \n', p1net.player.team_alive[0:prn].cpu().numpy())
			print('QA attack: \n', tmpqa[0:prn, :4].cpu().numpy())
			print('QA switch: \n', tmpqa[0:prn, 4:].cpu().numpy())
			print('QB attack: \n', tmpqb[0:prn, :4].cpu().numpy())
			print('QB switch: \n', tmpqb[0:prn, 4:].cpu().numpy())
			print('V:         \n', tmpv[0:prn].cpu().numpy())
		

		# do an eval epoch
		if (i % eval_epoch_every == eval_epoch_every - 1):

			# agent plays argmax of q function
			p1net.eval()
			for k in range(parallel_per_batch):
				p1s[k].evalmode = True
				p1s[k].warmup = False
				p1s[k].wins = 0

			p1wins_eval, p2wins_eval = 0, 0

			for j in range(batch_size):
				winner_strings = run_parallel_learning_episode(
					parallel_per_batch, p1s, p2s, p1net, formatid=formatid, player_team_size=player_team_size, verbose=False, train=True)

				for k in range(parallel_per_batch):
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
			for k in range(parallel_per_batch):
				p1s[k].evalmode = False

			p1winrate_eval = float(p1wins_eval) / float(p1wins_eval + p2wins_eval)
			p2winrate_eval = float(p2wins_eval) / float(p1wins_eval + p2wins_eval)

			max_eval_winrate = max(p1winrate_eval, max_winrate)

			print('\n[Epoch {:3d}: Evaluation] '.format(i))
			print('Player 1 | win rate : {:.4f} |  '.format(
				p1winrate_eval) + 'wins : {:4d}  '.format(p1wins_eval) + int(50 * p1winrate_eval) * '#')
			print('Player 2 | win rate : {:.4f} |  '.format(
				p2winrate_eval) + 'wins : {:4d}  '.format(p2wins_eval) + int(50 * p2winrate_eval) * '#')
			print('(Player 1 | {}-ave. train win rate : {:.4f})'.format(eval_epoch_every, sum(train_win_array[-(eval_epoch_every):]) / eval_epoch_every ))
			print()

			if(p1winrate_eval >= max_eval_winrate):
				# save model if eval win rate improved
				save_model(fstring, c, i, model=p1net, model_target=v_target_net, optimizer=optimizer)

			eval_win_array.append(p1winrate_eval)

			with open('output/' + fstring + '_train_win_rates_c=' + str(c) + '_ep=' + str(i) + '.csv', 'w') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(train_win_array)

			with open('output/' + fstring + '_eval_win_rates_c=' + str(c) + '_ep=' + str(i) + '.csv', 'w') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(eval_win_array)


	# end
	results = {
		'train_win_rates' : train_win_array,
		'eval_win_rates' : eval_win_array,
	}

	return p1s, p2s, optimizer, p1net, replay, results




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
	'''
	#determinism when running on cuda (needed to reproduce runs)
	if(DEVICE == 'cuda:0'):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	'''

	# parameters
	# state_embeddings must be divisible by 4 (for MultiHeadAttention heads=4)
	state_embedding_settings = {
		'move':        		{'embed_dim': 32, 'dict_size': neural_net_small_2.MAX_TOK_MOVE},
		'type':        		{'embed_dim': 16, 'dict_size': neural_net_small_2.MAX_TOK_TYPE},
		'condition':   		{'embed_dim': 8, 'dict_size': neural_net_small_2.MAX_TOK_CONDITION},
	    'move_category':    {'embed_dim': 8, 'dict_size': neural_net_small_2.MAX_TOK_MOVE_CATEGORY},
	}

	hidden_layer_settings = {
		'player' : 64,
		'opponent' : 64,
		'context' : 64,
		'pokemon_hidden' : 32,

	}

	fstring = 'run6v6'

	load_state = False
	load_fstring = 'run3v3_0_5'
	
	# game
	epochs = 100
	batch_size = 16
	parallel_per_batch = 64
	eval_epoch_every = 5
	formatid = 'gen5ou'

	gamma = 0.99
	lam = 0.95
	verbose = True

	# training
	alpha = 0.4
	warmup_epochs = 5  # random playing

	# experience replay	
	replay_size = 1e5 
	minibatch_size = 150

	train_update_iters = 150
	print_obj_every = 50

	# player 1 neural net (initialize target network the same)
	p1net = SmallDeePikachu2(
        state_embedding_settings,
        hidden_layer_settings,
		move_identity=True,
		layer_norm=True,
        dropout=0.0,
        attention=True)
	p1net = p1net.to(DEVICE)

	v_target_net = copy.deepcopy(p1net)
	v_target_net.to(DEVICE)

	replay = ExperienceReplay(size=int(replay_size), minibatch_size=minibatch_size)

	# agents
	p1s = [ParallelLearningAgent(
		id='p1', name='Red', size=MAX_GAME_LEN + 1, gamma=gamma, lam=lam, alpha=alpha) for _ in range(parallel_per_batch)]
	p2s = [RandomAgent(id='p2', name='Blue') for _ in range(parallel_per_batch)]

	# optimizer 
	lr = 0.0001 # 0.0004 (SAC paper recommendation)
	weight_decay = 1e-5
	optimizer = optim.Adam(p1net.parameters(), lr=lr, weight_decay=weight_decay)
	

	# load if intended
	if load_state:
		p1net, v_target_net, optimizer, c, loaded_epoch = load_model(
			load_fstring, model=p1net, model_target=v_target_net, optimizer=optimizer)
		starting_epoch = loaded_epoch + 1
	else:
		starting_epoch = 0 # don't change

	'''
	###################
	#  POKEMON ARENA  #
	###################
	''' 

	# sample teams: player_team_size in [1 .. 6]
	player_team_size = 6

	# run training epochs
	p1s, p2s, optimizer, p1net, replay, results = train_parallel_epochs(
		p1s=p1s, p2s=p2s, optimizer=optimizer, p1net=p1net, v_target_net=v_target_net, replay=replay,
		formatid=formatid, player_team_size=player_team_size,
		alpha=alpha, warmup_epochs=warmup_epochs, train_update_iters=train_update_iters,
		epochs=epochs, batch_size=batch_size, parallel_per_batch=parallel_per_batch, 
		starting_epoch=starting_epoch, eval_epoch_every=eval_epoch_every,
		gamma=gamma, lam=lam,
		fstring=fstring, tt_print=print_obj_every, verbose=verbose
	)

	with open('output/' + fstring + '_train_win_rates_' + str(c) + '_END.csv', 'w') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(results['train_win_rates'])

	with open('output/' + fstring + '_eval_win_rates_' + str(c) + '_END.csv', 'w') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(results['eval_win_rates'])



	# state_embedding_settings = {
	# 	'pokemon':     {'embed_dim': 32, 'dict_size': neural_net.MAX_TOK_POKEMON},
	# 	'move':        {'embed_dim': 16, 'dict_size': neural_net.MAX_TOK_MOVE},
	# 	'type':        {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_TYPE},
	# 	'move_type':   {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_MOVE_TYPE},
	# 	'ability':     {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_ABILITY},
	# 	'item':        {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_ITEM},
	# 	'condition':   {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_CONDITION},
	# 	'weather':     {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_WEATHER},
	# 	'alive':       {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_ALIVE},
	# 	'disabled':    {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_DISABLED},
	# 	'spikes':      {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_SPIKES},
	# 	'toxicspikes': {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_TOXSPIKES},
	# 	'fieldeffect': {'embed_dim': 4, 'dict_size': neural_net.MAX_TOK_FIELD},
	# }
