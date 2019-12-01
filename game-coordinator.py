
import time
import sys
import subprocess
import json
import pprint
import copy

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

    # adressed players
    if (type == 'sideupdate'):
        # next line is adressed player, o/w it's both players
        adressed_players = raw.pop(0).split('\n')[0]
    else:
        adressed_players = ['p1', 'p2']

    # handle all messages in sequence
    split_active = 'no' # if active, is set to either 'p1' or 'p2'

    for j, s__ in enumerate(raw):

        s_ = s__.split('\n')[0]
        s = s_.split('|')[1:]

       
        if s[0] == 'win':
            # we have a winner
            message = copy.deepcopy(MESSAGE['win'])
            # next to next line is json of game info
            _ = simulator.stdout.readline()
            line = simulator.stdout.readline()
            message['info_json'] = json.loads(line.split('\n')[0])
            obj = SimulatorMessage(type, adressed_players_ind, message, original_str=s_)
            message_objects.append(obj)
            break

        # first part of message is id (remove all special characters like `-`)
        id = s.pop(0) 
        id = id if id != '' else 'empty' 
        # convert '-minoraction' to 'minor_minoraction'
        id = 'minor_' + id[1:] if id[0] == '-' else id
        if id not in SIMULATOR_MESSAGE_IDS:
            raise ValueError('Unknown simulator message ID \'' + id + '\'')
        
        # get corresponding MESSAGE and fill values 
        # (important that field order in MESSAGE is the correct)
        message = copy.deepcopy(MESSAGE[id])

        # process special messages
        if id == 'request':
            request_dict = json.loads(s[0])
            message['request_dict'] = request_dict
            
        elif id == 'empty':
            pass
        else:
            # process all regular messages
            # first field of MESSAGE is always 'id' and doesn't need to be filled
            message_fields = list(message.keys())
            message_fields.pop(0) 
            if (len(message_fields) < len(s)):
                raise ValueError(
                    'Message by simulator and corresponding '
                    'MESSAGE object dont have the same number of fields '
                    '(not enough fields to be filled in Message)')
            
            # fill message object in order
            for i, field in enumerate(message_fields):
                if i <= len(s) - 1:
                    message[field] = s[i]
                else:
                    # not all information provided stream message so don't fill dict
                    break         
                        
        # regular case: just record the current message
        if not id == 'split':

            # create SimulatorMessage
            adressed_players_ind = adressed_players if split_active == 'no' else split_active
            obj = SimulatorMessage(type, adressed_players_ind, message, original_str=s_)
            message_objects.append(obj)

            # reset split flag (in case it was on)
            split_active = 'no'

        # don't record `split` messages per se as they only indicate the next recipient
        else:
            if not split_active == 'no':
                raise ValueError('split message flag should not '
                                 'be active when split appears')
            # next message is only visible to the specifically indicated player by split 
            split_active = message['player']

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

def filter_messages_by_player(player, messages):
    '''
    Filters messages that are only truly addressed to `player`
    '''
    player_messages = []
    for m in messages:
        if m.adressed_players == player or player in m.adressed_players:
            player_messages.append(m)
    return player_messages

def filter_messages_by_id(id, messages):
    '''
    Filters messages that have a certain id, e.g `faint`
    '''
    id_messages = []
    for m in messages:
        if m.message['id'] == id:
            id_messages.append(m)
    return id_messages

def retrieve_message_ids(messages):
    '''
    Returns list of all message ids in list of SimulatorMessages
    '''
    ids = set()
    for m in messages:
        ids.add(m.message['id'])
    return list(ids)

def get_player_request(player, messages):
    '''
    Searches through list of MESSAGEs and returns MOST RECENT request for player (p1 or p2)
    Returns None if no request for player
    '''
    request = None
    for m in messages:
        if m.adressed_players == player and m.message['id'] == 'request':
            request = m.message
    return request


def send_choice_to_simulator(player_action):
    '''
    Sends PlayerAction made by a player to the simulator
    '''
    player = player_action.player
    action_name = player_action.action['id']
    action_dict = player_action.action

    # this should produce string without \n defining the action after >p1 ..
    if action_name == 'team':
        # team 
        action_str = 'team ' + action_dict['teamspec']
    elif action_name == 'default':
        # default
        action_str = 'default'
    elif action_name == 'undo':
        # undo
        action_str = 'undo'
    elif action_name == 'move':
        # move
        action_str = 'move ' + action_dict['movespec']
    elif action_name == 'move_mega':
        # move mega
        action_str = 'move ' + action_dict['movespec'] + ' mega'
    elif action_name == 'move_zmove':
        # move zmove
        action_str = 'move ' + action_dict['movespec'] + ' zmove'
    elif action_name == 'switch':
        # switch
        action_str = 'switch ' + action_dict['switchspec']
    else:
        raise ValueError("Trying to send unspecified action to simulator")

    out = '>' + player + ' ' + action_str + '\n'
    simulator.stdin.write(out)
    simulator.stdin.flush()	

t_start = time.time()
SIMS = 50
for t in range(SIMS):

    '''
    START: Live code
    '''

    print(f'Simulation {t}')

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
    simulator.stdin.write('>start {"formatid":"gen1randombattle"}\n')
    simulator.stdin.write('>player p1 {"name":"' + player1.name +'"}\n')
    simulator.stdin.write('>player p2 {"name":"' + player2.name +'"}\n')
    simulator.stdin.flush()	
    MESSAGES_TURN = 3

    game = []

    # game flow
    turn = 0
    game_ended = False
    last_messages = game
    while not game_ended:

        '''
        Standard turn consists of 3 simulator messages
        - sideupdate p1 with request for p1
        - sideupdate p2 with request for p2
        - update about what happened last round

        Then, both p1 and p2 have to make a choice for next round

        If in the update we find out a pokemon fainted of p1 or p2,
        then the corresponding player has to make a second choice that turn (namely a switch)

        '''

        turn += 1
        # print(f'TURN  {turn}')

        # receive simulation updates and inform players
        last_messages = []
        for _ in range(MESSAGES_TURN):
            last_messages += receive_simulator_message()

            # check if game is over
            if 'win' in retrieve_message_ids(last_messages):
                game_ended = True
                break

        game += last_messages
        if game_ended:
            break

        player1.receive_game_update(filter_messages_by_player('p1', last_messages))
        player2.receive_game_update(filter_messages_by_player('p2', last_messages))
        message_ids = retrieve_message_ids(last_messages)

    
        # if message contains faint, request a single choice by the player with fainted pokemon 
        if 'faint' in message_ids:
            faint_messages = filter_messages_by_id('faint', last_messages)   
            
            # there can be two faint messages in case of `Explosion` move
            if len(faint_messages) == 2:
                # make sure actaully both pokemon fainted
                assert(faint_messages[0].message['pokemon'][0:2] != faint_messages[1].message['pokemon'][0:2])

            for faint_message in faint_messages:
                # first two characters of fainted pokemon describe the player
                fainted_player_id = faint_message.message['pokemon'][0:2]
                faint_request = get_player_request(fainted_player_id, last_messages)
                if fainted_player_id == 'p1':
                    faint_action = player1.process_request(faint_request)
                else:
                    faint_action = player2.process_request(faint_request)

                send_choice_to_simulator(faint_action)

        # if messages contain turn, regulat turn with a choice each player
        # if just request (special minor effect situations), then there are still requests for both  
        elif 'turn' or 'request' in message_ids: 
            request_p1 = get_player_request('p1', last_messages)
            request_p2 = get_player_request('p2', last_messages)

            action_p1 = player1.process_request(request_p1)
            action_p2 = player2.process_request(request_p2)

            send_choice_to_simulator(action_p1)
            send_choice_to_simulator(action_p2)


        # else not sure what situation we are in
        else:
            raise ValueError('Unknown game situation given MessageIDs: \n' + str(message_ids))


    # print results
    game_over_message = filter_messages_by_id('win', game)[0]
    # pprint.pprint(game_over_message.message['info_json'])
    # print(game_over_message.message['info_json'])

    # terminate game
    simulator.terminate()
    simulator.stdin.close()

t_end = time.time()
print('Ave. sim time: ' + str((t_end - t_start) / SIMS))