'''
team data to be read in
'''
import random

team1 = "Blastoise||focussash||scald, raindance,icebeam,rapidspin||85,85,85,85,85,85||||83|]Charizard||lifeorb||flamethrower,slash,tailwind,swordsdance||81,,85,85,85,85||,0,,,,||83|]Venusaur||choiceband||powerwhip,earthquake,sludgebomb,synthesis||85,85,85,85,85,85||||83|]Pikachu||lifeorb||thunder,irontail,surf,substitute||85,85,85,85,85,85||||79|]Mewtwo||choicespecs||flamethrower,icebeam,psychic,lightscreen||85,85,85,85,85,85||||83|]Ditto||choicescarf|H|transform,,,||85,85,85,85,85,85||||83|"
team2 = "Blastoise||focussash||raindance,scald,icebeam,toxic||85,85,85,85,85,85||||83|]Charizard||lifeorb||flareblitz,slash,tailwind,swordsdance||81,,85,85,85,85||,0,,,,||83|]Venusaur||choiceband||powerwhip,earthquake,sludgebomb,synthesis||85,85,85,85,85,85||||83|"
team3 = "Mewtwo||lifeorb||psychic,icebeam,shadowball,lightscreen||85,85,85,85,85,85||||83|"

i = random.randint(0, 287)
with open('construct_teams/team_folder/team_'+str(i)+'.txt', 'r') as file:
	team4 = file.read().replace('\n', '') 
