# coding=utf-8

import time
import sys
import subprocess
import json
import pprint
import copy
import teams_data

# import custom structures (like MESSAGE, ACTION) and all agents
from custom_structures import *
from agents import *

# TO RUN: python game_coordinator.py -p1 default -p2 default


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
        adressed_players = [raw.pop(0).split('\n')[0]]
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
                    '(not enough fields to be filled in Message): ' + str(message))
            
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
        if player in m.adressed_players:
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

def retrieve_message_ids_set(messages):
    '''
    Returns set of all message ids in list of SimulatorMessages
    '''
    ids = set()
    for m in messages:
        ids.add(m.message['id'])
    return list(ids)

def retrieve_message_ids_list(messages):
    '''
    Returns list of all message ids in list of SimulatorMessages
    '''
    ids = []
    for m in messages:
        ids.append(m.message['id'])
    return ids

def get_player_request(player, messages):
    '''
    Searches through list of MESSAGEs and returns MOST RECENT request for player (p1 or p2)
    Returns None if no request for player
    '''
    request = []
    for m in messages:
        if player in m.adressed_players and (m.message['id'] == 'request'):
            request.append(m.message)
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
    elif action_name == 'userspecified':
        # user specified move via command line
        action_str = action_dict['string']
    else:
        print('Unspecified')
        raise ValueError("Trying to send unspecified action to simulator: " + str(action_dict))

    out = '>' + player + ' ' + action_str + '\n'
    simulator.stdin.write(out)
    simulator.stdin.flush()	

def create_agents_from_argv(args):
    '''
    Parses command line arguments and initializes players accordingly
    '''
    options = args[1:]

    if not (len(options) == 0 or len(options) == 2 or len(options) == 4):
        raise ValueError('Invalid number of arguments passed to game_coordinator.py.')

    info1 = None
    info2 = None
    while options:
        tmp = options.pop(0)
        if tmp == '-p1':
            info1 = options.pop(0)
        elif tmp == '-p2':
            info2 = options.pop(0)
        else:
            raise ValueError('Invalid argument passed to game_coordinator.py')
    
    # use default if not provided
    infos = [info1 if info1 else 'default', info2 if info2 else 'default']
    names = ['Player 1', 'Player 2']
    ps = ['p1', 'p2']
    out = []
    for j in range(2):
        info, p, name = infos[j], ps[j], names[j]
        if info == 'default':
            player = DefaultAgent(p, name=name)
        elif info == 'random':
            player = RandomAgent(p, name=name)
        elif info == 'human':
            player = HumanAgent(p, name=name)
        else:
            raise ValueError('Agent type provided is not defined:' + p)
        out.append(player)
        
    print('Player 1:  ' + infos[0])
    print('Player 2:  ' + infos[1])
    return tuple(out)

if __name__ == '__main__':

    '''
    START: Live code
    Simulates one game.
    # python game_coordinator.py -p1 default -p2 default
    If not provied, use default agent
    '''

    # parse arguments and initialize players
    SIMS = 100
    for i in range(0, SIMS):
        print(f'Simulation # {i+1}/{SIMS}')

        player1, player2 = create_agents_from_argv(sys.argv)

        # opens: pokemon-showdown simulate-battle
        simulator = subprocess.Popen('./pokemon-showdown simulate-battle', 
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True)

        # start game 
        # simulator.stdin.write('>start {"formatid":"gen5randombattle"}\n')
        simulator.stdin.write('>start {"formatid":"gen5ou"}\n')
        #simulator.stdin.write('>player p1 {"name":"' + player1.name + '"' + ',"team":"' + teams_data.team1 +'" }\n')
        #simulator.stdin.write('>player p2 {"name":"' + player2.name + '"' + ',"team":"' + teams_data.team1 +'" }\n')
        simulator.stdin.write('>player p1 {"name":"' + player1.name + '" }\n')
        simulator.stdin.write('>player p2 {"name":"' + player2.name +'" }\n')
        
        simulator.stdin.flush()	

        game = []

        outstanding_requests = []

        # regular game flow
        while True:
        
            # receive a simulation update and inform players
            new_messages = receive_simulator_message()
            message_ids = retrieve_message_ids_set(new_messages)
            game += new_messages
            
            # for m in new_messages:
            #     if not m.message['id'] == 'request':
            #         print(m.original_str)
            #     else:
            #         pprint.pprint(m.message['request_dict'])
            # for m in new_messages:
            #     if m.message['id'] == 'error':
            #         print(m.original_str)


            # if at least one player is human print all messages (except requests)
            if isinstance(player1, HumanAgent):
                for m in filter_messages_by_player('p1', new_messages):
                    if not m.message['id'] == 'request':
                        print(m.original_str)
            if isinstance(player2, HumanAgent):
                for m in filter_messages_by_player('p2', new_messages):
                    if not m.message['id'] == 'request':
                        print(m.original_str)

            # check if game is over    
            if 'win' in message_ids:
                break

            # if there are requests, record them
            if 'request' in message_ids: 
                outstanding_requests += filter_messages_by_id('request', new_messages)
            
            # regular messages f this was a normal update, then send updates to players, and afterwards request move
            else:
                
                # 1) update players on new information
                player1.receive_game_update(filter_messages_by_player('p1', new_messages))
                player2.receive_game_update(filter_messages_by_player('p2', new_messages))

                # 2) request outstanding moves
                while outstanding_requests:
                    m_request = outstanding_requests.pop(0)
                    player = m_request.adressed_players
                    if len(player) != 1:
                        raise ValueError('requests should only be addressed to one player')
                    else:
                        player = player[0]

                    # only request an action if request is not 'wait' request
                    # (requesting move ('active') or switch ('forceSwitch') or team ('teamPreview'))
                    if not 'wait' in m_request.message['request_dict'].keys():
                        if player == 'p1':
                            action = player1.process_request(m_request)
                        else:
                            action = player2.process_request(m_request)

                        send_choice_to_simulator(action)
                

        # print results
        game_over_message = filter_messages_by_id('win', game)[0]
        # pprint.pprint(game_over_message.message['info_json'])
        #print(game_over_message.message['info_json'])

        # terminate game
        simulator.terminate()
        simulator.stdin.close()
        
