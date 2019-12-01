
import time
import sys
import subprocess
import json
from enum import Enum


''' 
This file needs to be in the same folder as "pokemon-showdown simulate-battle"


Notes on IO functions:

Use this to work only with strings and avoid all binaries
proc = subprocess.Popen('python repeater.py', 
    shell=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    universal_newlines=True)

proc.stdin.write(x): 
    x must be binary and contain a new line; can convert to binary using b'...' or x_str.encode()
    Ex.: proc.stdin.write(b'test 1 \n')
proc.stdout.readline():
    returns binary, which can be converted to string using .decode()
(if you set proc.universal_newlines=True, then all of the above binary vars become strings)

sys.stdout.write(x):
    x must be of type string and contain a new line
    Ex.: sys.stdout.write('repeater.py: bla 1 \n')
sys.stdin.readline():
    returns string


'''

PLAYER_IDS = ['p1', 'p2']
SIMULATOR_MESSAGE_TYPES = ['update', 'sideupdate', 'end']

class MESSAGE(Enum):
    '''
    This enum represents and defines all game messages
    Doc: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md

    Important invariants:
    1 - Enum name and its corresponding 'id' MUST BE THE SAME 
        (exception: empty message i.e. '|\n' has id 'empty')
    2 - The order of fields in the dict has to be the order of info received by the game stream
        (Example: player message stream is ...|<player>|<username>|<avatar>|<rating> 
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




# ''.join(e for e in string if e.isalnum())

# list of all ids defined in MESSAGE enum
SIMULATOR_MESSAGE_IDS =  [m.value['id'] if not m.name == 'empty' else 'empty' for m in MESSAGE]

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


def parse_simulator_message(raw):
    '''
    Parses the string of one full message by the simulator 
    Returns a list of SimulatorMessage objects, one object ``for every`` message 
    that is part of the full message (e.g. one update can contain many messages)
    '''
    
    message_objects = []

    # type
    type = raw.pop(0).split('\n')[0]
    if type not in SIMULATOR_MESSAGE_TYPES:
        raise ValueError('Unknown simulator message type \'' + type + '\'')
    # print('TYPE: ', type)

    # adressed players
    if (type == 'sideupdate'):
        # next line is adressed player, o/w it's both players
        adressed_players = raw.pop(0).split('\n')[0]
    else:
        adressed_players = ['p1', 'p2']

    # handle all messages in sequence
    split_active = 'no' # if active, is set to either 'p1' or 'p2'

    for s_ in raw:
        s = s_.split('\n')[0].split('|')[1:]

        # first part of message is id (remove all special characters like `-`)
        id = s.pop(0) 
        id = id if id != '' else 'empty' 
        id = ''.join(ch for ch in id if ch.isalnum())
        if id not in SIMULATOR_MESSAGE_IDS:
            raise ValueError('Unknown simulator message ID \'' + id + '\'')
        
        # get corresponding MESSAGE and fill values 
        # (important that field order in MESSAGE is the correct)
        message = MESSAGE[id]


        # process special messages
        if id == 'request':
            request_dict = json.loads(s[0])
            message.value['request_dict'] = request_dict
            
        elif id == 'empty':

            pass
            
        else:
            # process all regular messages
            # first field of MESSAGE is always 'id' and doesn't need to be filled
            message_fields = list(message.value.keys())
            message_fields.pop(0) 
            if (len(message_fields) < len(s)):
                raise ValueError(
                    'Message by simulator and corresponding '
                    'MESSAGE object dont have the same number of fields '
                    '(not enough fields to be filled in Message)')

            # fill message object in order
            for i, field in enumerate(message_fields):
                message.value[field] = s[i]

        # regular case: just record the current message
        if not id == 'split':

            # create SimulatorMessage
            adressed_players_ind = adressed_players if split_active == 'no' else split_active
            obj = SimulatorMessage(type, adressed_players_ind, message)
            message_objects.append(obj)

            # reset split flag (in case it was on)
            split_active = 'no'

        # don't record `split` messages per se as they only indicate the next recipient
        if id == 'split':
            if not split_active == 'no':
                raise ValueError('split message flag should not '
                                 'be active when split appears')
            # next message is only visible to the specifically indicated player by split 
            split_active = message.value['player']

    for m in message_objects:
        print(m)
    return message_objects

def receive_simulator_message():
    '''
    Live receives one full message by the simulator (ending with '\n\n')
    Return list of SimulatorMessage objects corresponding to full message string
    '''
    raw_message = []
    while (True):
        line = simulator.stdout.readline()
        if (line == '\n'):
            break
        raw_message.append(line)
    messages = parse_simulator_message(raw_message)
    return messages

'''
START: Live code
'''

# opens: pokemon-showdown simulate-battle
simulator = subprocess.Popen('./pokemon-showdown simulate-battle', 
    shell=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    universal_newlines=True)


# start game 
# (special as you need to read exactly 3 messages by simulator)
simulator.stdin.write('>start {"formatid":"gen1randombattle"}\n')
simulator.stdin.write('>player p1 {"name":"Scott"}\n')
simulator.stdin.write('>player p2 {"name":"Lars"}\n')
simulator.stdin.flush()	
START_MESSAGES = 3


game = []
for _ in range(START_MESSAGES):
    new_message = receive_simulator_message()
    game.append(new_message)


# terminate game
simulator.terminate()
simulator.stdin.close()



