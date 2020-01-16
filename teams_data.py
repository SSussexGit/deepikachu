'''
team data to be read in
'''
import random
import numpy as np

team1 = "Blastoise||focussash||scald, raindance,icebeam,rapidspin||85,85,85,85,85,85||||83|]Charizard||lifeorb||flamethrower,slash,tailwind,swordsdance||81,,85,85,85,85||,0,,,,||83|]Venusaur||choiceband||powerwhip,earthquake,sludgebomb,synthesis||85,85,85,85,85,85||||83|]Pikachu||lifeorb||thunder,irontail,surf,substitute||85,85,85,85,85,85||||79|]Mewtwo||choicespecs||flamethrower,icebeam,psychic,lightscreen||85,85,85,85,85,85||||83|]Ditto||choicescarf|H|transform,,,||85,85,85,85,85,85||||83|"
team2 = "Blastoise||focussash||raindance,scald,icebeam,toxic||85,85,85,85,85,85||||83|]Charizard||lifeorb||flareblitz,slash,tailwind,swordsdance||81,,85,85,85,85||,0,,,,||83|]Venusaur||choiceband||powerwhip,earthquake,sludgebomb,synthesis||85,85,85,85,85,85||||83|"
team3 = "Mewtwo||lifeorb||psychic,icebeam,shadowball,lightscreen||85,85,85,85,85,85||||83|"

TEAMCOUNT = 286
train_size = 257
train_teams_index = np.asarray([163,  35,  94, 215, 172, 150, 271,  46, 149, 206, 234, 189,  42,
       139, 240, 126,  45,  34,  24, 265,  75, 235, 141, 108,  79,  21,
        17,  83, 220, 227,  56,  82, 151,  12, 280, 187, 132, 115, 124,
       112, 253,  52,   7, 248,  86, 245, 101, 237, 239, 114,  25, 122,
       148,  97,  51, 211,  65, 213, 168,  55, 222,  47, 170, 219, 249,
        60,   1, 197, 182,  71, 138,  36, 247,  78, 125, 143, 262,  96,
       160,  90, 167, 242, 264, 270, 174, 185, 233, 229,  61, 231, 246,
       255, 186,  32,  30,  74, 260,  85, 154, 217,  81,   9,  89, 166,
         3, 144, 274,  70,  22,  37, 191, 230, 106,  14, 103,   4,  68,
        11, 110,  93, 250,   6, 119,  50, 203,  15,  31, 123,  44, 146,
       218, 256,   5, 109, 276,  92,  29, 169, 152, 224, 136,  49, 198,
       244, 113,  95,  69, 171, 130, 283, 121,  18, 212, 118, 258,  16,
       133, 278,  19, 210, 223,  28,  41, 161, 117, 120, 200, 156,  72,
        48, 285,  88,  27, 181, 195, 221, 140, 232, 155,  26, 135, 177,
       184,  77,  64,  76, 202, 282,  38,  23, 137, 105, 261, 142,  80,
       111,  20, 158, 165, 205,  67,  62, 145,  99, 236, 102,  87, 194,
       263, 277, 193, 266,  53,  39,  84,   0, 100,  66, 153, 127, 134,
        13, 190,  10, 226, 216, 180, 104, 178,  63, 204, 175, 228, 196,
         2, 176, 257, 188, 252, 241, 201, 159, 107,  57, 214, 207, 164,
       259, 173, 162, 254,  40, 281,  91, 225, 179,  58])

test_teams_index = np.asarray([147, 284, 8, 273,  54, 209, 243, 267, 251, 269,  73, 192, 208,
       157, 275, 268,  33, 128, 129, 116, 279, 131,  59,  43, 183, 238,
       272,  98, 199])

def get_random_team(N, train=True):
	'''
	Returns random team of size N
	'''
	N = min(N, 6)
	team = ''
	if(train):
		idx = train_teams_index[random.randint(0, train_size-1)]
	else:
		idx = test_teams_index[random.randint(0, (TEAMCOUNT-train_size)-1)]
	with open('construct_teams/team_folder/team_'+str(idx)+'.txt', 'r') as file:
		full_team_string = file.read().replace('\n', '')
		individual_pokemon_strings = full_team_string.split(']')
		idxs = np.random.choice(6, size=N, replace=False)
		for i in range(N):
			team += individual_pokemon_strings[idxs[i]]
			if not (i == N - 1):
				team += ']'

	return team
