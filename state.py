




# dummy variable to represent tokens
# each of these have to have an UNK and NON token 
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
	'level': 0,
	'item': item_token,
	'stats': {
		'max_hp': 0,
		'atk': 0,
		'def': 0,
		'spa': 0,
		'spd': 0,
		'spe': 0},
	'moves' : {
		0 : default_move_state,
		1 : default_move_state,
		2 : default_move_state,
		3 : default_move_state,
	}}

default_state = {
	'player' : {
		'active' : default_pokemon_state,
		'team' : {
			0 : default_pokemon_state,
			1 : default_pokemon_state,
			2 : default_pokemon_state,
			3 : default_pokemon_state,
			4 : default_pokemon_state,
			5 : default_pokemon_state,
		}
	},
	'opponent' : {
		'active' : default_pokemon_state, # UNK tokens for most things
		'team' : {
			0 : default_pokemon_state,
			1 : default_pokemon_state,
			2 : default_pokemon_state,
			3 : default_pokemon_state,
			4 : default_pokemon_state,
			5 : default_pokemon_state,
		}
	},
	'field' : {
		'weathertype' : weather_token,
		'weathersummoned' : 0,
		'terrain' : terrain_token,
		'terrainsummoned' : 0,
		'tailwind' : False,
		'tailwindtime' : 0,
		'tailwindopp' : False,
		'tailwindtime' : 0,
		'encore' : False,
		'encoretime' : 0,
		'encoreopp' : False,
		'encoreopptime' : 0,
		'taunt' : False,
		'taunttime' : 0,
		'tauntopp' : False,
		'tauntopptime' : 0,
		'torment' : False,
		'tormenttime' : 0,
		'tormentopp' : False,
		'tormentopptime' : 0,
		'twoturnmove' : False,
		'twoturnmovenum' : 0, #a move number for the move that is two turns long
		'twoturnmoveopp' : False,
		'twoturnmoveoppnum' : 0,
		'confusion' : False,
		'confusionopp' : False, 
	},
}

def game_name_to_dex_name(s):
	#makes a string lower-case and removes hyphens
	return s.lower().replace('-', '')

