
import time
import sys
import subprocess
import json

# import custom structures (like MESSAGE, ACTION)
from custom_structures import *
from state import *
from data_pokemon import *

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
                    pokemon_state = copy.deepcopy(default_pokemon_state)

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
                        pokemon_state['moves'][j] = move_data[move_string]['num']
                        #if the max pp of the move is 0 in state, it is turn 0 so set the max pp and then pp = max pp

                        j+=1

                    

                    #extract stat information 
                    for key in pokemon_state['stats']:
                        #handle max_hp seperately after because it is given in the condition
                        if key != 'max_hp':
                            pokemon_state['stats'][key] = pokemon_dict['stats'][key]

                    #extract hp, max hp and condition

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




                    #copy the pokemon into the team
                    self.state['player']['team'][i] = copy.deepcopy(pokemon_state)

                    #if active set your active pokemon's details to this
                    if(pokemon_dict['active'] == True):
                        #update the pp values and disabled status of moves

                        self.state['player']['active'] = self.state['player']['team'][i]


                    i+=1
                #if we didn't update 6 pokemon fill in remaining spots with default values
                while i < 5:
                    self.state['player']['team'][i] = copy.deepcopy(default_pokemon_state)
                    i+=1

        self.history += messages


    def process_request(self, request):
        '''
        Receives request sent by `pokemon-showdown simulate-game` and returns a PlayerAction
        '''

        choice = copy.deepcopy(ACTION['default'])

        return PlayerAction(self.id, choice)

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


