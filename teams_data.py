'''
team data to be read in
'''
import random
import numpy

team1 = "Blastoise||focussash||scald, raindance,icebeam,rapidspin||85,85,85,85,85,85||||83|]Charizard||lifeorb||flamethrower,slash,tailwind,swordsdance||81,,85,85,85,85||,0,,,,||83|]Venusaur||choiceband||powerwhip,earthquake,sludgebomb,synthesis||85,85,85,85,85,85||||83|]Pikachu||lifeorb||thunder,irontail,surf,substitute||85,85,85,85,85,85||||79|]Mewtwo||choicespecs||flamethrower,icebeam,psychic,lightscreen||85,85,85,85,85,85||||83|]Ditto||choicescarf|H|transform,,,||85,85,85,85,85,85||||83|"
team2 = "Blastoise||focussash||raindance,scald,icebeam,toxic||85,85,85,85,85,85||||83|]Charizard||lifeorb||flareblitz,slash,tailwind,swordsdance||81,,85,85,85,85||,0,,,,||83|]Venusaur||choiceband||powerwhip,earthquake,sludgebomb,synthesis||85,85,85,85,85,85||||83|"
team3 = "Mewtwo||lifeorb||psychic,icebeam,shadowball,lightscreen||85,85,85,85,85,85||||83|"

TEAMCOUNT = 286

def get_random_team(N):
	'''
	Returns random team of size N
	'''
	N = min(N, 6)
	team = ''
	idx = random.randint(0, TEAMCOUNT - 1) 
	with open('construct_teams/team_folder/team_'+str(idx)+'.txt', 'r') as file:
		full_team_string = file.read().replace('\n', '')
		individual_pokemon_strings = full_team_string.split(']')
		idxs = np.random.choice(6, size=6, replace=False)
		for i in range(N):
			team += individual_pokemon_strings[idxs[i]]
			if not (i == N - 1):
				team += ']'

	return team
