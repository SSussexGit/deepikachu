
import time
import sys
import subprocess
import json
from enum import Enum

# import custom structures (like MESSAGE, ACTION)
from custom_structures import *

class Agent:
    '''
    Class implementing a random-playing agent interacting with game-coordinator.py
    '''
    def __init__(self):
        
        self.history = []

    def receive_game_update(self, messages):
        '''
        Receives series of game updates and process them
        '''
        self.history += messages


    def process_request(self, request):
        '''
        Receives request sent by `pokemon-showdown simulate-game` and returns an ACTION
        '''
