
import time
import sys
import subprocess
import json
import random

# import custom structures (like MESSAGE, ACTION)
from custom_structures import *
from state import *
from data_pokemon import *

def get_valid_actions(state, message):
    '''
    From a current state and request message, get our valid actions 
    '''
    #start with all options and remove
    valid_list = [{'id':'move', 'movespec': '1'}, {'id':'move', 'movespec': '2'}, {'id':'move', 'movespec': '3'}, {'id':'move', 'movespec': '4'}, {'id':'switch', 'switchspec': '1'}, {'id':'switch', 'switchspec': '2'}, {'id':'switch', 'switchspec': '3'}, {'id':'switch', 'switchspec': '4'}, {'id':'switch', 'switchspec': '5'},  {'id':'switch', 'switchspec': '6'}]

    #check if in a position with forceSwitch (can switch to anything alive except active)

    #just check if forceswitch in the message since never false
    if ('forceSwitch' in message):
        #remove all moves
        valid_list = [s for s in valid_list if not (s['id'] == 'move')]

    #go through the pokemon to make updates
    i = 1
    for pokemon_dict_index in state['player']['team']:
        pokemon_dict = state['player']['team'][pokemon_dict_index]
        if (pokemon_dict['active'] == True):
            #remove switching to the thing that is already active
            s = {'id':'switch', 'switchspec': i}
            if(s in valid_list):
                valid_list.remove()
        i+=1

    #go through active moves and don't allow ones that are disabled or no pp. 
    i = 1
    for move_dict_index in state['player']['active']['moves']:
        move_dict = state['player']['active']['moves'][move_dict_index]

        if ((move_dict['pp'] == 0) or (move_dict['disabled'] == True)):
            #keep the move if it is called struggle
            if(move_dict['id']!= move_data['struggle']['num']):
                s = {'id':'move', 'movespec': i}
                if(s in valid_list):
                    valid_list.remove(s)
        i+=1

    return valid_list



        #check if something is active (if not must switch to something alive)

class DefaultAgent:
    '''
    Class implementing a default-playing agent interacting with game-coordinator.py
    id is either 'p1' or 'p2'
    '''
    def __init__(self, id, name='Ash'):
        self.id = id
        self.name = name
        self.history = []
        self.state = copy.deepcopy(default_state)

    def receive_game_update(self, messages):
        '''
        Receives series of game updates and process them
        '''
        for message in messages:
            #if you get a request completely update your side
            if(message.message['id'] == 'request'):
                i = 0
                for pokemon_dict in message.message['request_dict']['side']['pokemon']:
                    #construct the pokemon state and then add to the team
                    #copy the existing pokemon state, update it, add it in
                    pokemon_state = copy.deepcopy(self.state['player']['team'][i])

                    #extract pokemon token
                    pokemon_name_string = pokemon_dict['ident'].split(': ')[1]
                    pokemon_state['pokemon_id']  = pokedex_data[game_name_to_dex_name(pokemon_name_string)]['num']

                    #extract move information
                    j = 0
                    for move_string in pokemon_dict['moves']:
                        #there may be other special cases in addition to "return102"
                        if(move_string.startswith("return")):
                            move_string= "return"
                        if(move_string.startswith("frustration")):
                            move_string= "frustration"
                        pokemon_state['moves'][j]['id'] = move_data[move_string]['num']
                        #if the max pp of the move is 0 in state, it is turn 0 so set the max pp and then pp = max pp
                        if(pokemon_state['moves'][j]['maxpp'] == 0):
                            pokemon_state['moves'][j]['maxpp'] = move_data[move_string]['pp']
                            pokemon_state['moves'][j]['pp'] = move_data[move_string]['pp']
                        #if the type is not yet set, fill it in
                        if(pokemon_state['moves'][j]['type'] == None):
                            pokemon_state['moves'][j]['type'] = type_data[move_data[move_string]['type'].lower()]

                        j+=1

                    #extract stat information 
                    for key in pokemon_state['stats']:
                        #handle max_hp seperately after because it is given in the condition
                        if key != 'max_hp':
                            pokemon_state['stats'][key] = pokemon_dict['stats'][key]

                    #extract hp, max hp and condition
                    #first split on spaces and check length to see if painted or status.
                    
                    condition_list = pokemon_dict['condition'].split(' ')
                    if(len(condition_list) == 2):
                        if(condition_list[1] == 'fnt'):
                            pokemon_state['alive'] = False
                        if(condition_list[1] in status_data):
                            pokemon_state['condition'] = status_data[condition_list[1]]['num']

                    #then split first on '/' to get hp values
                    health_values = condition_list[0].split('/')
                    if(len(health_values) == 2):
                        pokemon_state['stats']['max_hp'] = health_values[1]
                    pokemon_state['hp'] = health_values[0]

                    #extract item information
                    item_string = pokemon_dict['item']
                    if(item_string == ''):
                        pokemon_state['item'] = 0 #if no item tag just put 0
                    else:   
                        pokemon_state['item'] = item_data[item_string]['num']
                    

                    #extract ability information
                    ability_string = pokemon_dict['baseAbility']
                    if(ability_string == ''):
                        pokemon_state['ability'] = 0 
                    else:   
                        pokemon_state['ability'] = ability_data[ability_string]['num']


                    #extract type information 
                    pokemon_type_list = pokedex_data[game_name_to_dex_name(pokemon_name_string)]['types']

                    pokemon_state['type1'] = type_data[pokemon_type_list[0].lower()]['num']
                    #check if the thing has two types
                    if(len(pokemon_type_list) == 2):
                        pokemon_state['type2'] = type_data[pokemon_type_list[1].lower()]['num']
                    else: 
                        pokemon_state['type2'] = None
                    


                    

                    #if active set your active pokemon's details to this
                    if(pokemon_dict['active'] == True):

                        self.state['player']['team']
                        #update the pp values and disabled status of moves
                        j = 0
                        if 'active' in message.message['request_dict']:
                            for move_dict in message.message['request_dict']['active'][0]['moves']:
                                #if the move is struggle  it has no 'pp' so need this if statement
                                if (('pp' in move_dict) and ('disabled' in move_dict)):
                                    pokemon_state['moves'][j]['pp'] =  move_dict['pp']
                                    pokemon_state['moves'][j]['disabled'] =  move_dict['disabled']
                                j+=1

                            pokemon_state['active'] = True
                            self.state['player']['active'] = copy.deepcopy(pokemon_state)
                        else:
                            pokemon_state['active'] = False


                    #copy the pokemon into the team
                    self.state['player']['team'][i] = copy.deepcopy(pokemon_state)
                    i+=1

                #if there is no active pokemon, set your active to the null case
                if 'active' not in message.message['request_dict']:
                    self.state['player']['active'] = default_pokemon_state
                #if we didn't update 6 pokemon fill in remaining spots with default values
                while i < 5:
                    self.state['player']['team'][i] = copy.deepcopy(default_pokemon_state)
                    i+=1
        #print(self.state)
        self.history += messages


    def process_request(self, request):
        '''
        Receives request sent by `pokemon-showdown simulate-game` and returns a PlayerAction
        '''
        message = request.message['request_dict']
        print(message)
        #first get our valid action space
        valid_actions = get_valid_actions(self.state, message)

        if (valid_actions == []):
            random_action = 'default'
        else:
            random_action = random.choice(valid_actions)
        #choice = copy.deepcopy(ACTION['default'])
        print(random_action)
        return PlayerAction(self.id, random_action)

class RandomAgent(DefaultAgent):
    '''
    Class implementing player choosing random (valid) moves
    '''
    def process_request(self, request):
        '''
        Receives request sent by `pokemon-showdown simulate-game` and returns a PlayerAction
        '''

        print('RANDOM AGENT')
        exit(0)

        choice = copy.deepcopy(ACTION['default'])

        return PlayerAction(self.id, choice)


