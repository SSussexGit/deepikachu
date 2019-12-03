
import time
import sys
import subprocess
import json

# import custom structures (like MESSAGE, ACTION)
from custom_structures import *

class DefaultAgent:
    '''
    Class implementing a default-playing agent interacting with game-coordinator.py
    id is either 'p1' or 'p2'
    '''
    def __init__(self, id, name='Ash'):
        self.id = id
        self.name = name
        self.history = []

    def receive_game_update(self, messages):
        '''
        Receives series of game updates and process them
        '''


        self.history += messages


    def process_request(self, request):
        '''
        Receives request sent by `pokemon-showdown simulate-game` and returns a PlayerAction
        '''

        choice = copy.deepcopy(ACTION['default'])

        return PlayerAction(self.id, choice)
