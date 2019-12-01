from enum import Enum


PLAYER_IDS = ['p1', 'p2']
SIMULATOR_MESSAGE_TYPES = ['update', 'sideupdate', 'end']

class MESSAGE(Enum):
    '''
    This enum represents and defines all game messages
    Doc: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

    Important invariants:
    1 - Enum name and its corresponding 'id' must be the same
        (exception: empty message i.e. '|\n' has id 'empty')
    2 - The order of fields in the dict has to be the order of info received by the game stream
        (Example: player message stream is |player|<player>|<username>|<avatar>|<rating> 
    '''

    # battle initialization
    split =         dict(id='split', player=None)
    player =        dict(id='player', player=None, username=None, avatar=None, rating=None)
    teamsize =      dict(id='teamsize', player=None, number=None)
    gametype =      dict(id='gametype', gametype=None)
    gen =           dict(id='gen', gennum=None)
    tier =          dict(id='tier', formatname=None)
    rule =          dict(id='rule', rule=None)
    rated =         dict(id='rated', message=None)
    clearpoke =     dict(id='clearpoke')
    start =         dict(id='start')
    poke =          dict(id='poke', player=None, details=None, item=None)

    # Battle progress
    empty =         dict(id=''), # empty line, i.e. '|'
    request =       dict(id='request', request_dict=None)
    inactive =      dict(id='inactive', message=None)
    inactiveoff =   dict(id='inactiveoff', message=None)
    upkeep =        dict(id='upkeep')
    turn =          dict(id='turn', number=None)
    win =           dict(id='win', user=None)
    tie =           dict(id='tie')
    error =         dict(id='error', message=None)

    # Major actions
    move =          dict(id='move', pokemon=None, move=None, target=None)
    switch =        dict(id='switch', pokemon=None, details=None, hp=None)
    drag =          dict(id='drag', pokemon=None, details=None, hp=None)
    detailschange = dict(id='detailschange', pokemon=None, details=None, hp=None)
    formechange =   dict(id='formechange', pokemon=None, species=None, hp=None)
    replace =       dict(id='replace', pokemon=None, details=None, hp=None)
    swap =          dict(id='swap', pokemon=None, position=None)
    cant =          dict(id='cant', pokemon=None, reason=None, move=None)
    faint =         dict(id='faint', pokemon=None)

    # Minor actions
    
    minor_fail =                dict(id='minor_fail', pokemon=None, action=None)
    minor_block =               dict(id='minor_block', pokemon=None, effect=None, move=None, attacker=None)
    minor_notarget =            dict(id='minor_notarget', pokemon=None)
    minor_miss =                dict(id='minor_miss', source=None, target=None)
    minor_damage =              dict(id='minor_damage', pokemon=None, hp=None)
    minor_heal =                dict(id='minor_heal', pokemon=None, hp=None)
    minor_sethp =               dict(id='minor_sethp', pokemon=None, hp=None)
    minor_status =              dict(id='minor_status', pokemon=None, status=None)
    minor_curestatus =          dict(id='minor_urestatus', pokemon=None, status=None)
    minor_cureteam =            dict(id='minor_cureteam', pokemon=None)
    minor_boost =               dict(id='minor_boost', pokemon=None, stat=None, amount=None)
    minor_unboost =             dict(id='minor_unboost', pokemon=None, stat=None, amount=None)
    minor_setboost =            dict(id='minor_setboost', pokemon=None, stat=None, amount=None)
    minor_swapboost =           dict(id='minor_swapboost', source=None, target=None, stats=None)
    minor_invertboost =         dict(id='minor_invertboost', pokemon=None)
    minor_clearboost =          dict(id='minor_clearboost', pokemon=None)
    minor_clearallboost =       dict(id='minor_clearallboost')
    minor_clearpositiveboost =  dict(id='minor_clearpositiveboost', target=None, pokemon=None, effect=None)
    minor_clearnegativeboost =  dict(id='minor_clearnegativeboost', pokemon=None)
    minor_copyboost =           dict(id='minor_copyboost', source=None, target=None)
    minor_weather =             dict(id='minor_weather', weather=None)
    minor_fieldstart =          dict(id='minor_fieldstart', condition=None)
    minor_fieldend =            dict(id='minor_fieldend', condition=None)
    minor_sidestart =           dict(id='minor_sidestart', side=None, condition=None)
    minor_sideend =             dict(id='minor_sideend', side=None, condition=None)
    minor_start =               dict(id='minor_start', pokemon=None, effect=None)
    minor_end =                 dict(id='minor_end', pokemon=None, effect=None)
    minor_crit =                dict(id='minor_crit', pokemon=None)
    minor_supereffective =      dict(id='minor_supereffective', pokemon=None)
    minor_resisted =            dict(id='minor_resisted', pokemon=None)
    minor_immune =              dict(id='minor_immune', pokemon=None)
    minor_item =                dict(id='minor_item', pokemon=None, item=None)
    minor_enditem =             dict(id='minor_enditem', pokemon=None, item=None)
    minor_ability =             dict(id='minor_ability', pokemon=None, ability=None)
    minor_endability =          dict(id='minor_endability', pokemon=None)
    minor_transform =           dict(id='minor_transform', pokemon=None, species=None)
    minor_mega =                dict(id='minor_mega', pokemon=None, megastone=None)
    minor_primal =              dict(id='minor_primal', pokemon=None)
    minor_burst =               dict(id='minor_burst', pokemon=None, species=None, item=None)
    minor_zpower =              dict(id='minor_zpower', pokemon=None)
    minor_zbroken =             dict(id='minor_zbroken', pokemon=None)
    minor_activate =            dict(id='minor_activate', effect=None,)
    minor_hint =                dict(id='minor_hint', message=None)
    minor_center =              dict(id='minor_center')
    minor_message =             dict(id='minor_message', message=None)
    minor_combine =             dict(id='minor_combine')
    minor_waiting =             dict(id='minor_waiting', source=None, target=None)
    minor_prepare =             dict(id='minor_prepare', attacker=None, move=None, defender=None)
    minor_mustrecharge =        dict(id='minor_mustrecharge', pokemon=None)
    minor_nothing =             dict(id='minor_nothing')
    minor_hitcount =            dict(id='minor_hitcount', pokemon=None, num=None)
    minor_singlemove =          dict(id='minor_singlemove', pokemon=None, move=None)
    minor_singleturn =          dict(id='minor_singleturn', pokemon=None, move=None)



# list of all ids defined in MESSAGE enum
SIMULATOR_MESSAGE_IDS =  [m.value['id'] if not m.name == 'empty' else 'empty' for m in MESSAGE]



class ACTION(Enum):
	'''
	This enum represents and defines all game choices for players(i.e. actions)
	Doc: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

	'''

	'''TEAMSPEC is list of pokemon slots e.g. team 213456 '''
	team =         		dict(teamspec=None)

	'''To auto-choose a decision. This will be the first possible legal choice'''
	default =        	dict()

	'''To cancel a previously-made choice. This can only be done if the another player needs to make a choice and hasn't done so'''
	undo =         		dict()

	'''A regular game choice. 
	MOVESPEC is a move name or 1-based move slot number
	SWITCHSPEC is a Pokémon nickname/species or 1-based slot number
	'''
	move =  			dict(movespec=None)
	move_mega =  		dict(movespec=None)
	move_zmove =  		dict(movespec=None)
	switch =  			dict(movespec=None)




class SimulatorMessage:
    '''
    Class that represents one full message by the simulator 
    '''
    def __init__(self, type, adressed_players, message):
        # (string from SIMULATOR_MESSAGE_TYPES)
        self.type = type

        # list of (string from PLAYERS)
        self.adressed_players = adressed_players

        # (dict from MESSAGE)
        self.message = message

    def __str__(self):
        # for printing
        s = (f'SimulatorMessage [{self.type}] to:   {str(self.adressed_players)} \n' + 
            f' | ID: {self.message.name}  | {self.message.value}')
        return s


class PlayerAction:
    '''
    Class that represents an action by a player
    '''
    def __init__(self, player, action):
        # `p1` or `p2`
        self.player = player

        # (dict from ACTION)
        self.action = action

    def __str__(self):
        # for printing
        s = (f'PlayerAction [' + self.player + ']\n' + 
            f' | ID: {self.action.name}  | {self.action.value}')
        return s


