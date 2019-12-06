# coding=utf-8
import copy

PLAYER_IDS = ['p1', 'p2']
SIMULATOR_MESSAGE_TYPES = ['update', 'sideupdate', 'end']

'''
This dict represents and defines all game messages
Doc: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

Important invariants:
1 - key and its corresponding value 'id' must be the same
	(exception: empty message i.e. '|\n' has id 'empty'),
2 - The order of fields in the dict has to be the order of info received by the game stream
	(Example: player message stream is |player|<player>|<username>|<avatar>|<rating> 
'''
MESSAGE = {

	# battle initialization
	'split' :         dict(id='split', player=None),
	'player' :        dict(id='player', player=None, username=None, avatar=None, rating=None),
	'teamsize' :      dict(id='teamsize', player=None, number=None),
	'gametype' :      dict(id='gametype', gametype=None),
	'gen' :           dict(id='gen', gennum=None),
	'tier' :          dict(id='tier', formatname=None),
	'rule' :          dict(id='rule', rule=None),
	'rated' :         dict(id='rated', message=None),
	'clearpoke' :     dict(id='clearpoke'),
	'start' :         dict(id='start'),
	'poke' :          dict(id='poke', player=None, details=None, item=None),

	# Battle progress
	'empty' :         dict(id=''), # empty line, i.e. '|'
	'request' :       dict(id='request', request_dict=None),
	'inactive' :      dict(id='inactive', message=None),
	'inactiveoff' :   dict(id='inactiveoff', message=None),
	'upkeep' :        dict(id='upkeep'),
	'turn' :          dict(id='turn', number=None),
	'win' :           dict(id='win', player=None, info_json=None),
	'tie' :           dict(id='tie'),
	'error' :         dict(id='error', message=None),
	'teampreview' :   dict(id='teampreview', message=None),

	# Major actions
	'move' :          dict(id='move', pokemon=None, move=None, target=None, additional_info_=None, additional_info_2=None),
	'switch' :        dict(id='switch', pokemon=None, details=None, hp=None, additional_info=None),
	'drag' :          dict(id='drag', pokemon=None, details=None, hp=None, additional_info=None),
	'detailschange' : dict(id='detailschange', pokemon=None, details=None, hp=None, additional_info=None),
	'formechange' :   dict(id='formechange', pokemon=None, species=None, hp=None, additional_info=None),
	'replace' :       dict(id='replace', pokemon=None, details=None, hp=None, additional_info=None),
	'swap' :          dict(id='swap', pokemon=None, position=None, additional_info=None),
	'cant' :          dict(id='cant', pokemon=None, reason=None, move=None, additional_info=None),
	'faint' :         dict(id='faint', pokemon=None, additional_info=None),
	

	# Minor actions
	# had to add ``, additional_info=None` to all since the showdown documentation of the fields
	# does not perfectly match the information actually received (often get more info by the simulator),
	'minor_fail' :                dict(id='minor_fail', pokemon=None, action=None, additional_info=None, additional_info2=None),
	'minor_block' :               dict(id='minor_block', pokemon=None, effect=None, move=None, attacker=None, additional_info=None, additional_info2=None),
	'minor_notarget' :            dict(id='minor_notarget', pokemon=None, additional_info=None, additional_info2=None),
	'minor_miss' :                dict(id='minor_miss', source=None, target=None, additional_info=None, additional_info2=None),
	'minor_damage' :              dict(id='minor_damage', pokemon=None, hp=None, attack=None, source=None, additional_info=None, additional_info2=None),
	'minor_heal' :                dict(id='minor_heal', pokemon=None, hp=None, move=None, source=None, additional_info=None, additional_info2=None),
	'minor_sethp' :               dict(id='minor_sethp', pokemon=None, hp=None, additional_info=None, additional_info2=None),
	'minor_status' :              dict(id='minor_status', pokemon=None, status=None, move=None, additional_info=None, additional_info2=None),
	'minor_curestatus' :          dict(id='minor_curestatus', pokemon=None, status=None, additional_info=None, additional_info2=None),
	'minor_cureteam' :            dict(id='minor_cureteam', pokemon=None, additional_info=None, additional_info2=None),
	'minor_boost' :               dict(id='minor_boost', pokemon=None, stat=None, amount=None, additional_info=None, additional_info2=None),
	'minor_unboost' :             dict(id='minor_unboost', pokemon=None, stat=None, amount=None, additional_info=None, additional_info2=None),
	'minor_setboost' :            dict(id='minor_setboost', pokemon=None, stat=None, amount=None, additional_info=None, additional_info2=None),
	'minor_swapboost' :           dict(id='minor_swapboost', source=None, target=None, stats=None, additional_info=None, additional_info2=None),
	'minor_invertboost' :         dict(id='minor_invertboost', pokemon=None, additional_info=None, additional_info2=None),
	'minor_clearboost' :          dict(id='minor_clearboost', pokemon=None, additional_info=None, additional_info2=None),
	'minor_clearallboost' :       dict(id='minor_clearallboost', additional_info=None, additional_info2=None),
	'minor_clearpositiveboost' :  dict(id='minor_clearpositiveboost', target=None, pokemon=None, effect=None, additional_info=None, additional_info2=None),
	'minor_clearnegativeboost' :  dict(id='minor_clearnegativeboost', pokemon=None, additional_info=None, additional_info2=None),
	'minor_copyboost' :           dict(id='minor_copyboost', source=None, target=None, additional_info=None, additional_info2=None),
	'minor_weather' :             dict(id='minor_weather', weather=None, from_ability=None, from_pokemon=None, additional_info2=None),
	'minor_fieldstart' :          dict(id='minor_fieldstart', condition=None, additional_info=None, additional_info2=None),
	'minor_fieldend' :            dict(id='minor_fieldend', condition=None, additional_info=None, additional_info2=None),
	'minor_sidestart' :           dict(id='minor_sidestart', side=None, condition=None, additional_info=None, additional_info2=None),
	'minor_sideend' :             dict(id='minor_sideend', side=None, condition=None, additional_info=None, additional_info2=None),
	'minor_start' :               dict(id='minor_start', pokemon=None, effect=None, additional_info=None, additional_info2=None, additional_info3=None),
	'minor_end' :                 dict(id='minor_end', pokemon=None, effect=None, additional_info=None, additional_info2=None),
	'minor_crit' :                dict(id='minor_crit', pokemon=None, additional_info=None, additional_info2=None),
	'minor_supereffective' :      dict(id='minor_supereffective', pokemon=None, additional_info=None, additional_info2=None),
	'minor_resisted' :            dict(id='minor_resisted', pokemon=None, additional_info=None, additional_info2=None),
	'minor_immune' :              dict(id='minor_immune', pokemon=None, additional_info=None, additional_info2=None),
	'minor_item' :                dict(id='minor_item', pokemon=None, item=None, additional_info=None, additional_info2=None),
	'minor_enditem' :             dict(id='minor_enditem', pokemon=None, item=None, additional_info=None, additional_info2=None),
	'minor_ability' :             dict(id='minor_ability', pokemon=None, ability=None, additional_info=None, additional_info2=None),
	'minor_endability' :          dict(id='minor_endability', pokemon=None, additional_info=None, additional_info2=None),
	'minor_transform' :           dict(id='minor_transform', pokemon=None, species=None, additional_info=None, additional_info2=None),
	'minor_mega' :                dict(id='minor_mega', pokemon=None, megastone=None, additional_info=None, additional_info2=None),
	'minor_primal' :              dict(id='minor_primal', pokemon=None, additional_info=None, additional_info2=None),
	'minor_burst' :               dict(id='minor_burst', pokemon=None, species=None, item=None, additional_info=None, additional_info2=None),
	'minor_zpower' :              dict(id='minor_zpower', pokemon=None, additional_info=None, additional_info2=None),
	'minor_zbroken' :             dict(id='minor_zbroken', pokemon=None, additional_info=None, additional_info2=None),
	'minor_activate' :            dict(id='minor_activate', pokemon=None, effect=None, additional_info=None, additional_info2=None),
	'minor_hint' :                dict(id='minor_hint', message=None, additional_info=None, additional_info2=None),
	'minor_center' :              dict(id='minor_center', additional_info=None, additional_info2=None),
	'minor_message' :             dict(id='minor_message', message=None, additional_info=None, additional_info2=None),
	'minor_combine' :             dict(id='minor_combine', additional_info=None, additional_info2=None),
	'minor_waiting' :             dict(id='minor_waiting', source=None, target=None, additional_info=None, additional_info2=None),
	'minor_prepare' :             dict(id='minor_prepare', attacker=None, move=None, defender=None, additional_info=None, additional_info2=None),
	'minor_mustrecharge' :        dict(id='minor_mustrecharge', pokemon=None, additional_info=None, additional_info2=None),
	'minor_nothing' :             dict(id='minor_nothing', additional_info=None, additional_info2=None),
	'minor_hitcount' :            dict(id='minor_hitcount', pokemon=None, num=None, additional_info=None, additional_info2=None),
	'minor_singlemove' :          dict(id='minor_singlemove', pokemon=None, move=None, additional_info=None, additional_info2=None),
	'minor_singleturn' :          dict(id='minor_singleturn', pokemon=None, move=None, additional_info=None, additional_info2=None),
	'minor_formechange' :         dict(id='minor_formechange', pokemon=None, species=None, hp=None, additional_info=None), # similar to `detailschange`
	'minor_anim' :				  dict(id='minor_anim', additional_info=None, additional_info2=None, additional_info3=None), # not in documentation
	'minor_fieldactivate':   	  dict(id='minor_fieldactivate', additional_info=None, additional_info2=None, additional_info3=None), # not in documentation

}


# list of all ids defined in MESSAGE dict
SIMULATOR_MESSAGE_IDS = list(MESSAGE.keys())


'''
This dict represents and defines all game choices for players(i.e. actions),
Doc: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

'''
ACTION = {
	
	# TEAMSPEC is list of pokemon slots e.g. team 213456 
	'team' :        dict(id='team', teamspec=None),

	# To auto-choose a decision. This will be the first possible legal choice
	'default' :     dict(id='default'),

	# To cancel a previously-made choice. This can only be done if the another player needs to make a choice and hasn't done so
	'undo' :        dict(id='undo'),

	# A regular game choice. 
	# MOVESPEC is a move name or 1-based move slot number
	# SWITCHSPEC is a Pokémon nickname/species or 1-based slot number
	'move' :  		dict(id='move', movespec=None),
	'move_mega' :  	dict(id='move_mega', movespec=None),
	'move_zmove' :  dict(id='move_zmove', movespec=None),
	'switch' :  	dict(id='switch', switchspec=None),
}



class SimulatorMessage:
	'''
	Class that represents one full message by the simulator 
	'''
	def __init__(self, type, adressed_players, message, original_str='NotProvided'):
		# (string from SIMULATOR_MESSAGE_TYPES),
		self.type = type

		# list of (string from PLAYERS),
		self.adressed_players = adressed_players

		# (dict from MESSAGE),
		self.message = message

		# original stream strings
		self.original_str = original_str

	def __str__(self):
		# for printing
		s = (f'SimulatorMessage [{self.type}] to:   {str(self.adressed_players),} \n' + 
			 f' | {self.message if self.message["id"] != "request" else "<request json>"}')
		return s


class PlayerAction:
    '''
    Class that represents an action by a player
    '''
    def __init__(self, player, action):
        # `p1` or `p2`
        self.player = player

        # (dict from ACTION),
        self.action = action

    def __str__(self):
        # for printing
        s = (f'PlayerAction [' + self.player + ']\n' + 
            f' | {self.action}')
        return s


