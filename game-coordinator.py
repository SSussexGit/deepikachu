
import time
import sys
import subprocess
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

PLAYERS = ['p1', 'p2']
SIMULATOR_MESSAGE_TYPES = ['update', 'sideupdate', 'end']
# SIMULATOR_MESSAGES__ = [
#     # Doc: https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md
#     # This list is intended to document the message prototype 
    
#     # Battle initialization
#     dict(id='split', request_json=None),
#     dict(id='player', player=None, username=None, avatar=None, rating=None),
#     dict(id='teamsize', player=None, number=None),
#     dict(id='gametype', gametype='singles'),
#     dict(id='gen', gennum=None),
#     dict(id='tier', formatname=None),
#     dict(id='rule', rule=None),
#     dict(id='rated', message=None),
#     dict(id='clearpoke'),
#     dict(id='start'),
#     dict(id='poke', player=None, details=None, item=None),

#     # Battle progress
#     dict(id=''), # empty line, i.e. '|'
#     dict(id='request', request_json=None),
#     dict(id='inactive', message=None),
#     dict(id='inactiveoff', message=None),
#     dict(id='upkeep'),
#     dict(id='turn', number=None),
#     dict(id='win', user=None),
#     dict(id='tie'),

#     # Major actions
#     dict(id='switch', request_json=None),
# ]
# SIMULATOR_MESSAGE_IDS = [d['id'] for d in SIMULATOR_MESSAGES__]

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
    split =         dict(id='split')
    player =        dict(id='player', player=None, username=None, avatar=None, rating=None)
    teamsize =      dict(id='teamsize', player=None, number=None)
    gametype =      dict(id='gametype', gametype='singles')
    gen =           dict(id='gen', gennum=None)
    tier =          dict(id='tier', formatname=None)
    rule =          dict(id='rule', rule=None)
    rated =         dict(id='rated', message=None)
    clearpoke =     dict(id='clearpoke')
    start =         dict(id='start')
    poke =          dict(id='poke', player=None, details=None, item=None)

    # Battle progress
    empty =         dict(id=''), # empty line, i.e. '|'
    request =       dict(id='request', request_json=None)
    inactive =      dict(id='inactive', message=None)
    inactiveoff =   dict(id='inactiveoff', message=None)
    upkeep =        dict(id='upkeep')
    turn =          dict(id='turn', number=None)
    win =           dict(id='win', user=None)
    tie =           dict(id='tie')

    # Major actions
    switch =        dict(id='switch', request_json=None)

# list of all ids defined in MESSAGE enum
SIMULATOR_MESSAGE_IDS =  [m.value['id'] if not m.name == 'empty' else 'empty' for m in MESSAGE]

class SimulatorMessage:
    '''
    Class that represents one full message by the simulator 
    '''
    def __init__(self, type, adressed_players, messages):
        # (string from SIMULATOR_MESSAGE_TYPES)
        self.type = type

        # list of (string from PLAYERS)
        self.adressed_players = adressed_players

        # list of (dict from MESSAGE)
        self.messages = messages


def parse_simulator_message(raw):
    '''
    Parses the string of one full message by the simulator 
    Returns a list of SimulatorMessage objects, one object ``for every`` message 
    that is part of the full message (e.g. one update can contain many messages)
    '''
    
    message_objects = []

    print('START MESSAGE')

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

    # handle all messages
    # print('ADRESSED PLAYERS: ', adressed_players)
    for s_ in raw:
        s = s_.split('\n')[0].split('|')[1:]

        # first part of message is id
        id = s.pop(0) 
        id = id if id != '' else 'empty' 
        if id not in SIMULATOR_MESSAGE_IDS:
            raise ValueError('Unknown simulator message ID \'' + id + '\'')
        
        # get corresponding MESSAGE and fill values 
        # (important that order in MESSAGE is the correct)
        message = MESSAGE[id]

       

        # special messages
        if id == 'request':
            
            # TODO json read in
            pass
        elif id == 'empty':

            pass
            
        elif id == 'split':

            # next two messages have different adressed_players
            # TODO

            pass

        elif id == 'switch':

            pass

        else:
            # all regular messages

            message_fields = list(message.value.keys())
            # first field is always 'id' and doesn't need to be filled
            message_fields.pop(0) 
            if (len(message_fields) != len(s)):
                raise ValueError(
                    'Message by simulator and corresponding '
                    'MESSAGE object dont have the same number of fields')
            
            # fill message object in order
            for i, field in enumerate(message_fields):
                message.value[field] = s[i]

        message_objects.append(message)
        print('Message: ' + repr(message))

   
    print('END MESSAGE')
    print()


    return raw

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



