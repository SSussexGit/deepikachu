

import copy
import numpy as np


# dummy variable to represent tokens
# each of these have to have an UNK and NON token 
#0 means unknown. 1 means known but empty
pokemon_token = 0
type_token =  0 
move_token = 0 
move_type_token = 0 
ability_token = 0 
item_token = 0 
weather_token = 0
terrain_token = 0
condition_token = 0
twoturnnum_token = 0
damage_category_token = 0

EMPTY = 1

# Next
# write function (used inside agent): game messages -> state
# this function needs to update online the state based on incoming messages (inside agent.receive_game_update)
# 
# agent.receive_game_update: 
# 	1) update the state
# agent.process_request:    
# 	1) work out valid actions (for masking at the end)
# 	2) use the current state and passing it through the policy function to output an action


#to add: damage, priority, physical/special, 
default_move_state = {
	'disabled': False,
	'moveid': move_token, #change in agent code
	'maxpp': 0,
	'pp': 0,
	'movetype' : move_type_token,
	'category' : damage_category_token,
	'accuracy' : 0,
	'priority' : 0, #priority anywhere from -6 to 6 like stat boosts
}

#to add: weight
default_pokemon_state = {
	'pokemon_id' : pokemon_token,
	'pokemontype1' : type_token,
	'pokemontype2' : type_token,
	'active': False, 
	'baseAbility': ability_token,
	'condition': EMPTY,
	'alive' : False,
	'hp': 0, 
	'level': 100,
	'item': item_token,
	'stats': {
		'max_hp': 0,
		'atk': 0,
		'def': 0,
		'spa': 0,
		'spd': 0,
		'spe': 0}, 
	'moves' : {
		0 : copy.deepcopy(default_move_state),
		1 : copy.deepcopy(default_move_state),
		2 : copy.deepcopy(default_move_state),
		3 : copy.deepcopy(default_move_state),
	}}

default_state = {
	'player' : {
		'active' : copy.deepcopy(default_pokemon_state),
		'boosts': {
			'atk': 0,
			'def': 0,
			'spa': 0,
			'spd': 0,
			'spe': 0, #just use the base stat with modifier applied. Also keep these as catagoricals
			'accuracy': 0, #XKCD we map to the modified
			'evasion': 0},
		'team' : {
			0 : copy.deepcopy(default_pokemon_state),
			1 : copy.deepcopy(default_pokemon_state),
			2 : copy.deepcopy(default_pokemon_state),
			3 : copy.deepcopy(default_pokemon_state),
			4 : copy.deepcopy(default_pokemon_state),
			5 : copy.deepcopy(default_pokemon_state),
		}
	},
	'opponent' : {
		'active' : copy.deepcopy(default_pokemon_state), 
		'boosts': {
			'oppatk': 0,
			'oppdef': 0,
			'oppspa': 0,
			'oppspd': 0,
			'oppspe': 0,
			'oppaccuracy': 0,
			'oppevasion': 0},
		'team' : {
			0 : copy.deepcopy(default_pokemon_state),
			1 : copy.deepcopy(default_pokemon_state),
			2 : copy.deepcopy(default_pokemon_state),
			3 : copy.deepcopy(default_pokemon_state),
			4 : copy.deepcopy(default_pokemon_state),
			5 : copy.deepcopy(default_pokemon_state),
		}
	},
	'field' : {
		'weather' : EMPTY,
		'weather_time' : 0,
		'terrain' : EMPTY,
		'terrain_time' : 0,
		'trickroom' : False,
		'trickroom_time' : 0, 
		'tailwind' : False,
		'tailwind_time' : 0,
		'tailwindopp' : False,
		'tailwindopp_time' : 0,
		'encore' : False,
		'encore_time' : 0,
		'encoreopp' : False,
		'encoreopp_time' : 0,
		'seed' : False,
		'seedopp' : False,
		'sub' : False,
		'subopp' : False,
		'taunt' : False,
		'taunt_time' : 0,
		'tauntopp' : False,
		'tauntopp_time' : 0,
		'torment' : False,
		'torment_time' : 0,
		'tormentopp' : False,
		'tormentopp_time' : 0,
		'twoturnmove' : False,
		'twoturnmoveid' : EMPTY, #a move number for the move that is two turns long
		'twoturnmoveopp' : False,
		'twoturnmoveoppid' : EMPTY,
		'confusion' : False,
		'confusionopp' : False, 
		'spikes' : 0, #int from 0 to 3 inclusive
		'spikesopp' : 0, 
		'toxicspikes' : 0, #int from 0 to 2 inclusive
		'toxicspikesopp' : 0, 
		'stealthrock' : False, 
		'stealthrockopp' : False, 
		'reflect' : False,
		'reflect_time' : 0,
		'reflectopp' : False,
		'reflectopp_time' : 0,
		'lightscreen' : False,
		'lightscreen_time' : 0,
		'lightscreenopp' : False,
		'lightscreenopp_time' : 0,
	},
}


#to add: weight
default_pokemon_state = {
	'pokemon_id' : pokemon_token,
	'pokemontype1' : type_token,
	'pokemontype2' : type_token,
	'active': False, 
	'baseAbility': ability_token,
	'condition': EMPTY,
	'alive' : False,
	'hp': 0, 
	'level': 100,
	'item': item_token,
	'stats': {
		'max_hp': 0,
		'atk': 0,
		'def': 0,
		'spa': 0,
		'spd': 0,
		'spe': 0}, 
	'moves' : {
		0 : copy.deepcopy(default_move_state),
		1 : copy.deepcopy(default_move_state),
		2 : copy.deepcopy(default_move_state),
		3 : copy.deepcopy(default_move_state),
	}}


#to add: damage, priority, physical/special, 
def create_2D_state(size):
	default_move_state2D = {
		'disabled': np.full((size), False, dtype=bool), 
		'moveid': np.full((size), move_token, dtype=int), #change in agent code
		'maxpp': np.full((size), 0, dtype=int),
		'pp': np.full((size), 0, dtype=int),
		'movetype' : np.full((size), move_type_token, dtype=int),
		'category' : np.full((size), damage_category_token, dtype=int),
		'accuracy' : np.full((size), 0, dtype=int),
		'priority' : np.full((size), 0, dtype=int), #priority anywhere from -6 to 6 like stat boosts
	}

	#to add: weight
	default_pokemon_state2D = {
		'pokemon_id' : np.full((size), pokemon_token, dtype=int),
		'pokemontype1' : np.full((size), type_token, dtype=int),
		'pokemontype2' : np.full((size), type_token, dtype=int),
		'active': np.full((size), False, dtype=bool), 
		'baseAbility': np.full((size), ability_token, dtype=int),
		'condition': np.full((size), EMPTY, dtype=int),
		'alive' : np.full((size), False, dtype=bool),
		'hp': np.full((size), 0, dtype=np.float32), 
		'level': np.full((size), 100, dtype=int),
		'item': np.full((size), item_token, dtype=int),
		'stats': {
			'max_hp': np.full((size), 0, dtype=np.float32),
			'atk': np.full((size), 0, dtype=np.float32),
			'def': np.full((size), 0, dtype=np.float32),
			'spa': np.full((size), 0, dtype=np.float32),
			'spd': np.full((size), 0, dtype=np.float32),
			'spe': np.full((size), 0, dtype=np.float32)}, 
		'moves' : {
			0 : copy.deepcopy(default_move_state2D),
			1 : copy.deepcopy(default_move_state2D),
			2 : copy.deepcopy(default_move_state2D),
			3 : copy.deepcopy(default_move_state2D),
		}}

	default_state2D = {
		'player' : {
			'active' : copy.deepcopy(default_pokemon_state2D),
			'boosts': {
				'atk': np.full((size), 0, dtype=int),
				'def': np.full((size), 0, dtype=int),
				'spa': np.full((size), 0, dtype=int),
				'spd': np.full((size), 0, dtype=int),
				'spe': np.full((size), 0, dtype=int), #just use the base stat with modifier applied. Also keep these as catagoricals
				'accuracy': np.full((size), 0, dtype=int), #XKCD we map to the modified
				'evasion': np.full((size), 0, dtype=int)},
			'team' : {
				0 : copy.deepcopy(default_pokemon_state2D),
				1 : copy.deepcopy(default_pokemon_state2D),
				2 : copy.deepcopy(default_pokemon_state2D),
				3 : copy.deepcopy(default_pokemon_state2D),
				4 : copy.deepcopy(default_pokemon_state2D),
				5 : copy.deepcopy(default_pokemon_state2D),
			}
		},
		'opponent' : {
			'active' : copy.deepcopy(default_pokemon_state2D), 
			'boosts': {
				'oppatk': np.full((size), 0, dtype=int),
				'oppdef': np.full((size), 0, dtype=int),
				'oppspa': np.full((size), 0, dtype=int),
				'oppspd': np.full((size), 0, dtype=int),
				'oppspe': np.full((size), 0, dtype=int),
				'oppaccuracy': np.full((size), 0, dtype=int),
				'oppevasion': np.full((size), 0, dtype=int)},
			'team' : {
				0 : copy.deepcopy(default_pokemon_state2D),
				1 : copy.deepcopy(default_pokemon_state2D),
				2 : copy.deepcopy(default_pokemon_state2D),
				3 : copy.deepcopy(default_pokemon_state2D),
				4 : copy.deepcopy(default_pokemon_state2D),
				5 : copy.deepcopy(default_pokemon_state2D),
			}
		},
		'field' : {
			'weather' : np.full((size), EMPTY, dtype=int),
			'weather_time' : np.full((size), 0, dtype=int),
			'terrain' : np.full((size), EMPTY, dtype=int),
			'terrain_time' : np.full((size), 0, dtype=int),
			'trickroom' : np.full((size), False, dtype=bool),
			'trickroom_time' : np.full((size), 0, dtype=int), 
			'tailwind' : np.full((size), False, dtype=bool),
			'tailwind_time' : np.full((size), 0, dtype=int),
			'tailwindopp' : np.full((size), False, dtype=bool),
			'tailwindopp_time' : np.full((size), 0, dtype=int),
			'encore' : np.full((size), False, dtype=bool),
			'encore_time' : np.full((size), 0, dtype=int),
			'encoreopp' : np.full((size), False, dtype=bool),
			'encoreopp_time' : np.full((size), 0, dtype=int),
			'seed' : np.full((size), False, dtype=bool),
			'seedopp' : np.full((size), False, dtype=bool),
			'sub' : np.full((size), False, dtype=bool),
			'subopp' : np.full((size), False, dtype=bool),
			'taunt' : np.full((size), False, dtype=bool),
			'taunt_time' : np.full((size), 0, dtype=int),
			'tauntopp' : np.full((size), False, dtype=bool),
			'tauntopp_time' : np.full((size), 0, dtype=int),
			'torment' : np.full((size), False, dtype=bool),
			'torment_time' : np.full((size), 0, dtype=int),
			'tormentopp' : np.full((size), False, dtype=bool),
			'tormentopp_time' : np.full((size), 0, dtype=int),
			'twoturnmove' : np.full((size), False, dtype=bool),
			'twoturnmoveid' : np.full((size), EMPTY, dtype=int), #a move number for the move that is two turns long
			'twoturnmoveopp' : np.full((size), False, dtype=bool),
			'twoturnmoveoppid' : np.full((size), EMPTY, dtype=int),
			'confusion' : np.full((size), False, dtype=bool),
			'confusionopp' : np.full((size), False, dtype=bool), 
			'spikes' : np.full((size), 0, dtype=int), #int from 0 to 3 inclusive
			'spikesopp' : np.full((size), 0, dtype=int), 
			'toxicspikes' : np.full((size), 0, dtype=int), #int from 0 to 2 inclusive
			'toxicspikesopp' : np.full((size), 0, dtype=int), 
			'stealthrock' : np.full((size), False, dtype=bool), 
			'stealthrockopp' : np.full((size), False, dtype=bool), 
			'reflect' : np.full((size), False, dtype=bool),
			'reflect_time' : np.full((size), 0, dtype=int),
			'reflectopp' : np.full((size), False, dtype=bool),
			'reflectopp_time' : np.full((size), 0, dtype=int),
			'lightscreen' : np.full((size), False, dtype=bool),
			'lightscreen_time' : np.full((size), 0, dtype=int),
			'lightscreenopp' : np.full((size), False, dtype=bool),
			'lightscreenopp_time' : np.full((size), 0, dtype=int),
		},
	}
	return copy.deepcopy(default_state2D)

def game_name_to_dex_name(s):
	#makes a string lower-case and removes hyphens
	return s.lower().replace('-', '').replace('. ', '').replace("'", "").replace('*', '').replace(" ", "")

