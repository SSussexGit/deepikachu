
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
    Work from the message rather than the state since message has things like being stuck into outrage
    '''
    #start with all options and remove
    valid_list = [
        {'id':'move', 'movespec': '1'}, 
        {'id':'move', 'movespec': '2'}, 
        {'id':'move', 'movespec': '3'}, 
        {'id':'move', 'movespec': '4'}, 
        {'id':'switch', 'switchspec': '1'}, 
        {'id':'switch', 'switchspec': '2'}, 
        {'id':'switch', 'switchspec': '3'}, 
        {'id':'switch', 'switchspec': '4'}, 
        {'id':'switch', 'switchspec': '5'},  
        {'id':'switch', 'switchspec': '6'},
    ]

    # check if we are in the team preview stage
    # completely disjoint set of options, so return list right here and now
    if ('teamPreview' in message):
        # Note: in theory could provide order like `412356`
        # but only the active pokemon is really a choice (here: 4)
        # and simulator supports just specifying first pokemon
        teamsize = int(message['maxTeamSize'])
        return [{'id':'team', 'teamspec': str(s)} for s in range(1, teamsize + 1)]

    # check if in a position with forceSwitch (can switch to anything alive except active)
    #just check if forceswitch in the message since never false
    if ('forceSwitch' in message):
        #remove all moves
        valid_list = [s for s in valid_list if not (s['id'] == 'move')]

    # go through the pokemon to make updates to valid moves
    i = 1
    for pokemon_dict in message['side']['pokemon']:
        if ((pokemon_dict['active'] == True) or (pokemon_dict['condition'] == '0 fnt')):
            #remove switching to the thing that is already active
            s = {'id':'switch', 'switchspec': str(i)}
            if(s in valid_list):
                valid_list.remove(s)

        i+=1

    #if something has less than 4 moves remove options. bit hacky
    
    if('active' in message):
        for j in range(0, 4-len(message['active'][0]['moves'])):
            s = {'id':'move', 'movespec': str(4-j)}
            if(s in valid_list):
                valid_list.remove(s)

        #if the active thing is trapped remove all switch options
        if('trapped' in message['active'][0] or'maybeTrapped' in message['active'][0]):
            valid_list = [s for s in valid_list if not (s['id'] == 'switch')]

        #go through active moves and don't allow ones that are disabled or no pp. 
        i = 1
        for move_dict in message['active'][0]['moves']:
            if ('pp' in move_dict): #if it doesn't have a pp term it is probably struggle
                if ((move_dict['pp'] == 0) or (move_dict['disabled'] == True)):
                    #keep the move if it is called struggle
                    if(move_dict['id']!= 'struggle'):
                        s = {'id':'move', 'movespec': str(i)}
                        if(s in valid_list):
                            valid_list.remove(s)
            i+=1
    else:
        #if no active pokemon, no move options
        for j in range(0, 4):
            s = {'id':'move', 'movespec': str(4-j)}
            if(s in valid_list):
                valid_list.remove(s)



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

    def request_update(self, message):
        i = 0
        for pokemon_dict in message['request_dict']['side']['pokemon']:
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
            if(len(condition_list) > 1):
                for condition in condition_list[1:]:
                    if(condition_list[1] == 'fnt'):
                        pokemon_state['alive'] = False
                    
                    #status confusion handled as special case terrain effect
                    for status_key in status_data:
                        if((status_key in condition_list[1]) and (status_key != 'confusion')):
                            pokemon_state['condition'] = status_data[status_key]['num']
                        else:
                            pokemon_state['condition'] = status_data[status_key]['num']
                    if("confusion" in condition_list):
                        self.state['field']['confusion'] = True
                    else:
                        self.state['field']['confusion'] = False
                    

            #then split first on '/' to get hp values
            health_values = condition_list[0].split('/')
            if(len(health_values) == 2):
                pokemon_state['stats']['max_hp'] = health_values[1]
            pokemon_state['hp'] = health_values[0]
            
            #set alive status
            if(health_values[0] != '0'):
                pokemon_state['alive'] = True
            else: 
                pokemon_state['alive'] = False

            #extract item information
            item_string = pokemon_dict['item']
            if(item_string == ''):
                pokemon_state['item'] = 0 #if no item tag just put 0
            else:   
                pokemon_state['item'] = item_data[item_string]['num']
            

            #extract ability information
            ability_string = pokemon_dict['baseAbility']
            if((ability_string == '') or (ability_string == None)):
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
                if 'active' in message['request_dict']:
                    for move_dict in message['request_dict']['active'][0]['moves']:
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
        if 'active' not in message['request_dict']:
            self.state['player']['active'] = default_pokemon_state
        #if we didn't update 6 pokemon fill in remaining spots with default values
        while i < 5:
            self.state['player']['team'][i] = copy.deepcopy(default_pokemon_state)
            i+=1
        return

    def impute_pokemon(self, message, pokemon_string):
        '''
        adds the pokemon name info to the opponent's team and returns the location it was added
        also adds the type and base stats
        '''
        pokemon_location = None
        none_list = []

        for pokemon_dict_index in self.state['opponent']['team']:
            pokemon_dict = copy.deepcopy(self.state['opponent']['team'][pokemon_dict_index])
            if(pokemon_dict['pokemon_id'] == pokedex_data[game_name_to_dex_name(pokemon_string)]['num']):
                pokemon_location = pokemon_dict_index
            if(pokemon_dict['pokemon_id'] == None):
                none_list.append(pokemon_dict_index)
            #make all the active status false
            self.state['opponent']['team'][pokemon_dict_index]['active'] = False

        #if already in the team then just update at that location 
        if pokemon_location != None:
            self.state['opponent']['team'][pokemon_location]['active'] = True 
            
        else:
            if(len(none_list) == 0):
                raise ValueError("Opponent's pokemon not seen in their team and team is full")
            pokemon_location = none_list[0]
            self.state['opponent']['active'] = self.state['opponent']['team'][pokemon_location]
            self.state['opponent']['team'][pokemon_location]['pokemon_id'] = pokedex_data[game_name_to_dex_name(pokemon_string)]['num']
            self.state['opponent']['team'][pokemon_location]['active'] = True 
            #if not in the team yet need to fill up the slots with everything
            types_list = pokedex_data[game_name_to_dex_name(pokemon_string)]['types']
            
            if(len(types_list)==2):
                self.state['opponent']['team'][pokemon_location]['type2'] = type_data[types_list[1].lower()]['num']
            
            self.state['opponent']['team'][pokemon_location]['type1'] = type_data[types_list[0].lower()]['num']

        #upon switching in or being in team preview the thing must be alive
        self.state['opponent']['team'][pokemon_location]['alive'] = True

        #add the base stats on
        for key in self.state['opponent']['team'][pokemon_location]['stats']:
            #handle max_hp seperately after because it is given in the condition
            if key == 'max_hp':
                self.state['opponent']['team'][pokemon_location]['stats'][key] = pokedex_data[game_name_to_dex_name(pokemon_string)]['baseStats']['hp']
            else:
                self.state['opponent']['team'][pokemon_location]['stats'][key] = pokedex_data[game_name_to_dex_name(pokemon_string)]['baseStats'][key]
        
        return pokemon_location

    def handle_poke_messages(self, message):
        pokemon_string = message['details'].split(',')[0]

        pokemon_location = self.impute_pokemon(message, pokemon_string)

        #set the things hp to be 100
        self.state['opponent']['team'][pokemon_location]['hp'] = 100

        return

    def get_pokemon_index(self, pokemon_string, player = 'opponent'):
        pokemon_location = None
        for pokemon_dict_index in self.state[player]['team']:
            pokemon_dict = copy.deepcopy(self.state[player]['team'][pokemon_dict_index])
            if(pokemon_dict['pokemon_id'] == pokedex_data[game_name_to_dex_name(pokemon_string)]['num']):
                pokemon_location = pokemon_dict_index
            if(pokemon_dict['pokemon_id'] == None):
                none_list.append(pokemon_dict_index)
        if(pokemon_location == None):
            raise ValueError("could not find pokemon in opponents team: " + pokemon_string)
        return pokemon_location

    def player_specific_update(self, message):
        '''
        extracts the player and pokemon from a pokemon field
        '''
        subject = message['pokemon']
        #extract into player and pokemon
        player_pokemon = subject.split(': ') #format is 'playerid: [pokemonname]'
        #extract the player id from the player part
        player_id = player_pokemon[0][:2]
        pokemon_name = player_pokemon[1]

        #update from a 'poke' message if we have teampreview
        #only need to update some things if it effects the opponent
        if(player_id != self.id):
            #assume it always updates the active pokemon
            #find the active pokemon and update it
          
            if(message['id'] == 'faint'):
                for pokemon_dict_index in self.state['opponent']['team']:
                    pokemon_dict = copy.deepcopy(self.state['opponent']['team'][pokemon_dict_index])
                    if(pokemon_dict['active'] == True):
                        pokemon_dict['alive'] = False
                        pokemon_dict['hp'] = 0

            #dealing with switching
            if message['id'] == 'switch' or message['id'] == "drag":
                #first check if the pokemon is in the state yet
                #retain a list of where Nones are and a value for where the pokemon is
                pokemon_location = None
                none_list = []

                pokemon_location = self.impute_pokemon(message, pokemon_name)
                #need to update the hp, status
                #split hp on the space, then on the /
                hp_condition = message['hp'].split(' ')
                minmaxhp = hp_condition[0].split('/')
                self.state['opponent']['team'][pokemon_location]['hp'] = minmaxhp[0]
                #if it has a status condition get it in there
                if(len(hp_condition) == 2):
                    if(hp_condition != 'fnt'):
                        self.state['opponent']['team'][pokemon_location]['condition'] = hp_condition[1]
                else: 
                    self.state['opponent']['team'][pokemon_location]['condition'] = None

                #copy into the active slot since it was switched in
                self.state['opponent']['active'] = copy.deepcopy(self.state['opponent']['team'][pokemon_location]) 
                
                #if they just switched they cannot have confusion
                self.state['field']['confusionopp'] = False

                #upon switch reset all the boosts
                for stat_string in self.state['opponent']['boosts']:
                    self.state['opponent']['boosts'][stat_string] = 0

            #minordamage to update hp
            if message['id'] == 'minordamage' or message['id'] == 'minorheal':
                pokemon_location = self.get_pokemon_index(pokemon_name)
                #need to update the hp, status
                #split hp on the space, then on the /
                hp_condition = message['hp'].split(' ')
                minmaxhp = hp_condition[0].split('/')
                self.state['opponent']['team'][pokemon_location]['hp'] = minmaxhp[0]
                #if it has a status condition get it in there
                if(len(hp_condition) == 2):
                    if(hp_condition != 'fnt'):
                        self.state['opponent']['team'][pokemon_location]['condition'] = hp_condition[1]
                else: 
                    self.state['opponent']['team'][pokemon_location]['condition'] = None

                if(message['id'] == 'minorheal'):
                    if('move' in message):
                        if('item' in message['move']): #if word item is in the message
                            item_string = message['move'].split(': ')[-1]#extract the item from the heal message and impute in the item slot
                            self.state['opponent']['team'][pokemon_location]['item'] = item_data[game_name_to_dex_name(item_string)]['num']

                if(message['id'] == 'minordamage'):
                    if('move' in message):
                        if('item' in message['move']): #if word item is in the message
                            item_string = message['move'].split(': ')[-1]#extract the item from the heal message and impute in the item slot
                            self.state['opponent']['team'][pokemon_location]['item'] = item_data[game_name_to_dex_name(item_string)]['num']

            #minor boost and unboost to update stats
            #maybe in 'stats for the opponent side just store base stats and have another dict that stores boosts as +1 etc' upon switch set all boosts to 0

            #when 'move' is used impute into their movepool

            #would be helpful to get in info about taunt, encore, torment

            #handle minor status message
            if(message['id'] == 'minorstatus'):
                pokemon_location = self.get_pokemon_index(pokemon_name)
                status_string = message['status']
                if (status_string == 'confusion'):
                    self.state['field']['confusionopp'] = True
                else:
                    self.state['opponent']['team'][pokemon_location]['status'] = status_data[status_string]['num']

            #handle minorstart for confusion induced by moves like outrage ending
            if(message['id'] == 'minorstart'):
                effect_string = message['effect']
                if (effect_string == 'confusion'):
                    self.state['field']['confusionopp'] = True

            #handle minorend for confusion
            if(message['id'] == 'minorend'):
                effect_string = message['effect']
                if (effect_string == 'confusion'):
                    self.state['field']['confusionopp'] = False

            #handle curestatus
            if(message['id'] == 'minor_curestatus'):
                pokemon_location = self.get_pokemon_index(pokemon_name)
                status_string = message['status']
                if (status_string == 'confusion'):
                    self.state['field']['confusionopp'] = False
                else:
                    self.state['opponent']['team'][pokemon_location]['status'] = None

            #handle cureteam which heals status of whole team
            if(message['id'] == 'minor_curestatus'):
                for pokemon_index in self.state['opponent']['team']:
                    self.state['opponent']['team'][pokemon_index]['status'] = None

            #handles boosts
            if(message['id'] in ['minor_boost', 'minor_unboost', 'minor_setboost']):
                pokemon_location = self.get_pokemon_index(pokemon_name)
                stat_string = message['stat']
                amount_int = int(message['amount'])
                if(message['id'] == 'minor_boost'):
                    self.state['opponent']['boosts'][stat_string] = min(self.state['opponent']['boosts'][stat_string] + amount_int, 6)
                if(message['id'] == 'minor_unboost'):
                    self.state['opponent']['boosts'][stat_string] = max(self.state['opponent']['boosts'][stat_string] - amount_int, -6)
                if(message['id'] == 'minor_setboost'):
                    self.state['opponent']['boosts'][stat_string] = amount_int
            if(message['id'] in ['minor_clearboost', 'minor_clearnegativeboost', 'minor_clearpositiveboost', 'minor_invertboost', 'minor_copyboost']):
                if(message['id'] == 'minor_clearboost'):
                    for stat_string in self.state['opponent']['boosts']:
                        self.state['opponent']['boosts'][stat_string] = 0
                if(message['id'] == 'minor_clearnegativeboost'):
                    for stat_string in self.state['opponent']['boosts']:
                        if self.state['opponent']['boosts'][stat_string] < 0:
                            self.state['opponent']['boosts'][stat_string] = 0
                if(message['id'] == 'minor_clearpositiveboost'):
                    for stat_string in self.state['opponent']['boosts']:
                        if self.state['opponent']['boosts'][stat_string] < 0:
                            self.state['opponent']['boosts'][stat_string] = 0
                if(message['id'] == 'minor_invertboost'):
                    for stat_string in self.state['player']['boosts']:
                        self.state['player']['boosts'][stat_string] = -self.state['player']['boosts'][stat_string]
                if(message['id'] == 'minor_copyboost'):
                    for stat_string in self.state['player']['boosts']:
                        self.state['player']['boosts'][stat_string] = self.state['opponent']['boosts'][stat_string]

            #if the pokemon of interest is active, update the active slot
            pokemon_location = self.get_pokemon_index(pokemon_name)
            if self.state['opponent']['team'][pokemon_location]['active']:
                self.state['opponent']['active'] = copy.deepcopy(self.state['opponent']['team'][pokemon_location])
        else:
            #for yourself

            #reset some things at switch-in
            #dealing with switching
            if message['id'] == 'switch' or message['id'] == "drag":

                #if they just switched they cannot have confusion
                self.state['field']['confusion'] = False

                #upon switch reset all the boosts
                for stat_string in self.state['player']['boosts']:
                    self.state['player']['boosts'][stat_string] = 0

            #handle boosts
            if(message['id'] in ['minor_boost', 'minor_unboost', 'minor_setboost']):
                stat_string = message['stat']
                amount_int = int(message['amount'])
                if(message['id'] == 'minor_boost'):
                    self.state['player']['boosts'][stat_string] = min(self.state['player']['boosts'][stat_string] + amount_int, 6)
                if(message['id'] == 'minor_unboost'):
                    self.state['player']['boosts'][stat_string] = max(self.state['player']['boosts'][stat_string] - amount_int, -6)
                if(message['id'] == 'minor_setboost'):
                    self.state['player']['boosts'][stat_string] = amount_int
            if(message['id'] in ['minor_clearboost', 'minor_clearnegativeboost', 'minor_clearpositiveboost', 'minor_invertboost', 'minor_copyboost']):
                if(message['id'] == 'minor_clearboost'):
                    for stat_string in self.state['player']['boosts']:
                        self.state['player']['boosts'][stat_string] = 0
                if(message['id'] == 'minor_clearnegativeboost'):
                    for stat_string in self.state['player']['boosts']:
                        if self.state['player']['boosts'][stat_string] < 0:
                            self.state['player']['boosts'][stat_string] = 0
                if(message['id'] == 'minor_clearpositiveboost'):
                    for stat_string in self.state['player']['boosts']:
                        if self.state['player']['boosts'][stat_string] < 0:
                            self.state['player']['boosts'][stat_string] = 0
                if(message['id'] == 'minor_invertboost'):
                    for stat_string in self.state['player']['boosts']:
                        self.state['player']['boosts'][stat_string] = -self.state['player']['boosts'][stat_string]
                if(message['id'] == 'minor_copyboost'):
                    for stat_string in self.state['player']['boosts']:
                        self.state['player']['boosts'][stat_string] = self.state['opponent']['boosts'][stat_string]


            #handle minorstart for confusion induced by moves like outrage ending
            if(message['id'] == 'minorstart'):
                effect_string = message['effect']
                if (effect_string == 'confusion'):
                    self.state['field']['confusion'] = True

            #handle minorend for confusion
            if(message['id'] == 'minorend'):
                effect_string = message['effect']
                if (effect_string == 'confusion'):
                    self.state['field']['confusion'] = False

        #sort clearallboost and swapboost
        if(message['id'] == 'minor_swapboost'):
            for stat_string in message['stats']:
                temp_value = self.state['opponent']['boosts'][stat_string]
                self.state['opponent']['boosts'][stat_string] = self.state['player']['boosts'][stat_string]
                self.state['player']['boosts'][stat_string] = self.state['opponent']['boosts'][stat_string]
        if(message['id'] == 'minor_clearallboost'):
            for stat_string in message['stats']:
                self.state['opponent']['boosts'][stat_string] = 0
                self.state['player']['boosts'][stat_string] = 0


        return

    def field_effect_update(self, message):
        if(message['id'] == 'minor_weather'):
            weather_string = game_name_to_dex_name(message['weather'].lower())
            if((weather_string != None) and (weather_string != 'none')):
            	self.state['field']['weathertype'] = weather_data[weather_string]['num']
            else:
            	self.state['field']['weathertype'] = None
            #now need to also add in how many turns the weather been up for
        #handle terrains

        #handle imputing the ability if the opponent set terain (not really needed)

        #handle trick room with minor_field and minor_fieldend

        #handle confusion with minor_start
        return

    def receive_game_update(self, messages):
        '''
        Receives series of game updates and process them
        Can move this into seperate functions if needed
        '''
        for message in messages:
            if(message.message['id'] == 'poke'):
                if(message.message['player'] != self.id):
                    #if a 'poke' message about the opponent, fill in their team
                    self.handle_poke_messages(message.message)
            #if its another kind of request update the field or opponent's side
            elif('pokemon' in message.message):
                self.player_specific_update(message.message)
                pass
            else:
                #if it's none of the above it pertains to a field effect

                self.field_effect_update(message.message)
                #pass
            
        self.history += messages

    def process_request(self, request):
        '''
        Receives request sent by `pokemon-showdown simulate-game` and returns a PlayerAction
        '''
        # Note: `default` also works for teamPreview stage
        #update state space
        self.request_update(request.message)
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
        #update state space
        self.request_update(request.message)
        message = request.message['request_dict']
        
        #first get our valid action space
        valid_actions = get_valid_actions(self.state, message)

        if (valid_actions == []):
            random_action = copy.deepcopy(ACTION['default'])
        else:
            random_action = random.choice(valid_actions)

        return PlayerAction(self.id, random_action)

class HumanAgent(DefaultAgent):
    '''
    Class implementing player choosing random (valid) moves
    '''
    def __action_dict_to_str(self, d):
        action_name = d['id']
        if action_name == 'team':
            s = 'team ' + d['teamspec']
        elif action_name == 'default':
            s = 'default'
        elif action_name == 'undo':
            s = 'undo'
        elif action_name == 'move':
            s = 'move ' + d['movespec']
        elif action_name == 'move_mega':
            s = 'move ' + d['movespec'] + ' mega'
        elif action_name == 'move_zmove':
            s = 'move ' + d['movespec'] + ' zmove'
        elif action_name == 'switch':
            s = 'switch ' + d['switchspec']
        else:
            raise ValueError("Unspecified action in parsing function: " + action_name)
        return s #"\'" + s + "\'"

    def __pretty(self, d, indent=0):
        gap = ' '
        line = 20 * '-'
        for key, value in d.items():
            if indent == 2:
                # after a certain depth just print a line 
                # s.t. its not too long
                print(gap * indent + "{:<12}".format(str(key)) + '\t' + str(value))
            else:
                if isinstance(value, list):
                    print(gap * indent + "{:<12}".format(str(key)))
                    for j, el in enumerate(value):
                        print(gap * indent + line + "  " + str(j + 1) + "  " + line)
                        if isinstance(el, dict):
                            self.__pretty(el, indent+1)
                        else:
                            print(gap * (indent+1) + str(el))
                elif isinstance(value, dict):
                    print(gap * indent + "{:<12}".format(str(key)))
                    self.__pretty(value, indent+1)
                else:
                    print(gap * indent + "{:<12}".format(str(key)) + '\t' + str(value))

    def process_request(self, request):
        '''
        Receives request sent by `pokemon-showdown simulate-game` and returns a PlayerAction 
        as specified by human command line input
        '''

        message = request.message['request_dict']
        # pretty print request for human readability
        print('\n======  CHOICE REQUEST   ======\n')
        self.__pretty(message, indent=0)
        print(f'\n\nVALID OPTIONS: {str()}')
        valid_actions_str = [self.__action_dict_to_str(d) for d in get_valid_actions(self.state, message)]
        for s in valid_actions_str:
            print('  ' + s)
        print('\nEnter move:')
        while (True):
            # request move until valid choice
            userstring = sys.stdin.readline()
            if userstring.split('\n')[0] in valid_actions_str:
                break
            print('Invalid move. Enter move:')
        print('\n======  GAME PROGRESS  ======\n')

        action = copy.deepcopy(ACTION['userspecified'])
        action['string'] = userstring
        return PlayerAction(self.id, action)



