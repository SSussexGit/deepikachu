

import copy


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
		'active' : copy.deepcopy(default_pokemon_state), # UNK tokens for most things
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
		'tailwind_time' : 0,
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

def game_name_to_dex_name(s):
	#makes a string lower-case and removes hyphens
	return s.lower().replace('-', '').replace('. ', '').replace("'", "").replace('*', '').replace(" ", "")

