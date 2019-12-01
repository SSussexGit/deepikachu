
import time
import sys
import subprocess
import json
from enum import Enum

# import custom structures (like MESSAGE, ACTION) and all agents
from custom_structures import *
from agents import *


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
        # convert '-minoraction' to 'minor_minoraction'
        id = 'minor_' + id if id[0] == '-' else id
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

def filter_messages(player, messages):
    '''
    Filters messages that are only truly addressed to `player`
    '''
    player_messages = []
    for m in messages:
        if m.adressed_players == player or player in m.adressed_players:
            player_messages.append(m)
    return player_messages

def get_player_request(player, messages):
    '''
    Searches through list of MESSAGEs and returns request for player (p1 or p2)
    '''
    for m in messages:
        if m.adressed_players == player and m.message.value['id'] == 'request':
            return m.message.value
    raise ValueError('No request for player found in list of messages.')


def send_choice_to_simulator(player_action):
    '''
    Sends PlayerAction made by a player to the simulator
    '''
    player = player_action.player
    action_name = player_action.action.name
    action_dict = player_action.action.value

    # this should produce string without \n defining the action after >p1 ..
    if action_name == ACTION.team.name:
        # team 
        action_str = 'team ' + action_dict['teamspec']
    elif action_name == ACTION.default.name:
        # default
        action_str = 'default'
    elif action_name == ACTION.undo.name:
        # undo
        action_str = 'undo'
    elif action_name == ACTION.move.name:
        # move
        action_str = 'move ' + action_dict['movespec']
    elif action_name == ACTION.move_mega.name:
        # move mega
        action_str = 'move ' + action_dict['movespec'] + ' mega'
    elif action_name == ACTION.move_zmove.name:
        # move zmove
        action_str = 'move ' + action_dict['movespec'] + ' zmove'
    elif action_name == ACTION.switch.name:
        # switch
        action_str = 'switch ' + action_dict['switchspec']
    else:
        raise ValueError("Trying to send unspecified action to simulator")

    out = '>' + player + ' ' + action_str + '\n'
    simulator.stdin.write(out)
    simulator.stdin.flush()	




'''
START: Live code
'''

print('Starting game simulation with two players.')

# initializes players
player1 = DefaultAgent('p1', name='Scott')
player2 = DefaultAgent('p2', name='Lars')

# opens: pokemon-showdown simulate-battle
simulator = subprocess.Popen('./pokemon-showdown simulate-battle', 
    shell=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    universal_newlines=True)


# start game 
# (special as you need to read exactly 3 messages by simulator)
simulator.stdin.write('>start {"formatid":"gen1randombattle"}\n')
simulator.stdin.write('>player p1 {"name":"' + player1.name +'"}\n')
simulator.stdin.write('>player p2 {"name":"' + player2.name +'"}\n')
simulator.stdin.flush()	
START_MESSAGES = 3

game = []
for _ in range(START_MESSAGES):
    new_messages = receive_simulator_message()
    game += new_messages

player1.receive_game_update(filter_messages('p1', game))
player2.receive_game_update(filter_messages('p2', game))


for m in game:
    print(m)


# game flow
game_ended = False
last_messages = game
while not game_ended:

    print('Waiting for player choices ...')
    # every player makes a choice
    request_p1 = get_player_request('p1', last_messages)
    request_p2 = get_player_request('p2', last_messages)

    # print('Player 1 request:')
    # print(request_p1)
    # print('Player 2 request:')
    # print(request_p2)

    action_p1 = player1.process_request(request_p1)
    action_p2 = player2.process_request(request_p2)

    print(action_p1)
    print(action_p2)

    send_choice_to_simulator(action_p1)
    send_choice_to_simulator(action_p2)



    print('Player choices sent.')


    # receive simulation results and inform players
    last_messages = receive_simulator_message()
    player1.receive_game_update(filter_messages('p1', last_messages))
    player2.receive_game_update(filter_messages('p2', last_messages))
    game.append(last_messages)


    for m in last_messages:
        print(m)

    break
    


# terminate game
simulator.terminate()
simulator.stdin.close()



