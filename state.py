

import copy


# dummy variable to represent tokens
# each of these have to have an UNK and NON token 
# if unknown use None, if has no value for this element use 0 (ie item is knocked off)
pokemon_token = None
type_token = None 
move_token = None 
move_type_token = None 
ability_token = None 
item_token = None 
weather_token = None
terrain_token = None
condition_token = None

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
	'id': move_token,
	'maxpp': 0,
	'pp': 0,
	'type' : move_type_token,
	'disabled' : False #can only be disabled if pokemon is active
}

#to add: weight
default_pokemon_state = {
	'pokemon_id' : pokemon_token,
	'type1' : type_token,
	'type2' : type_token,
	'active': False, 
	'baseAbility': ability_token,
	'condition': condition_token,
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
			'spe': 0,
			'accuracy': 0,
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
			'atk': 0,
			'def': 0,
			'spa': 0,
			'spd': 0,
			'spe': 0,
			'accuracy': 0,
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
	'field' : {
		'weather' : weather_token,
		'weather_time' : 0,
		'terrain' : terrain_token,
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
		'twoturnmovenum' : None, #a move number for the move that is two turns long
		'twoturnmoveopp' : False,
		'twoturnmoveoppnum' : None,
		'confusion' : False,
		'confusionopp' : False, 
		'spikes' : 0, #int from 0 to 3 inclusive
		'spiesopp' : 0, 
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

