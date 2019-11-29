
import time
import sys
import subprocess


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


def receive_simulator_message():
    '''
    Receives one full message by the simulator (ending with '\n\n')
    '''
    messages = []
    while (True):
        message = simulator.stdout.readline()
        if (message == '\n'):
            break
        # last two characters are '\n' so ignore them
        messages.append(message[:-1].split("|"))
    # first string will be empty as message starts with |
    return messages[1:] 



# opens: pokemon-showdown simulate-battle
simulator = subprocess.Popen('./pokemon-showdown simulate-battle', 
    shell=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    universal_newlines=True)


# start game 
# (different as you need to read exactly 3 messages by simulator)
simulator.stdin.write('>start {"formatid":"gen7randombattle"}\n')
simulator.stdin.write('>player p1 {"name":"Scott"}\n')
simulator.stdin.write('>player p2 {"name":"Lars"}\n')
simulator.stdin.flush()	
START_MESSAGES = 3


game = []
for _ in range(START_MESSAGES):
    new_message = receive_simulator_message()

    # sys.stderr.write('game-coordinator RECEIVED: ' + str(new_message))
    # sys.stderr.flush()

    game.append(new_message)

for s in game:
    for sub in s:
        print('\t' + str(sub))
    print()

# NEXT: splitting doesn't work yet (i'm remvoing some info). not always just | but is both \n and |

# AFTER THAT: add player classes

# while (True):
#     message = simulator.stdout.readline()

#     # report message
#     sys.stderr.write('game-coordinator RECEIVED: ' + message)
#     sys.stderr.flush()

#     if (message == '\n'):
#         break

    
# while (True):
#     message = simulator.stdout.readline()

#     # report message
#     sys.stderr.write('game-coordinator RECEIVED: ' + message)
#     sys.stderr.flush()

#     if (message == '\n'):
#         break



# while (True):
#     message = simulator.stdout.readline()

#     # report message
#     sys.stderr.write('game-coordinator RECEIVED: ' + message)
#     sys.stderr.flush()

#     if (message == '\n'):
#         break


# =---------

# # start game 
# simulator.stdin.write('>start {"formatid":"gen7randombattle"}\n')




# simulator.stdin.write('>player p1 {"name":"Alice"}\n>player p2 {"name":"Bob"}\n')
# simulator.stdin.flush()	

# sys.stderr.write('Wrote player p1\n')
# sys.stderr.flush()


# # receive lines by simulator until EOM ('\n\n')
# while (True):
#     message = simulator.stdout.readline()

#     # report message
#     sys.stderr.write('game-coordinator RECEIVED: ' + message)
#     sys.stderr.flush()

#     if (message == '\n'):
#         break

# simulator.stdin.write('>player p2 {"name":"Bob"}\n')
# simulator.stdin.flush()	

# sys.stderr.write('Wrote player p2\n')
# sys.stderr.flush()


# while (True):
#     message = simulator.stdout.readline()

#     # report message
#     sys.stderr.write('game-coordinator RECEIVED: ' + message)
#     sys.stderr.flush()

#     if (message == '\n'):
#         break

# =---------


# player_messages = [
# 	# '>player p1 {"name":"Alice"}',
# 	# '>player p2 {"name":"Bob"}',
# ]


# for player_message in player_messages:

#     # send message to simulator
#     simulator.stdin.write(player_message)
#     simulator.stdin.write('\n')
#     simulator.stdin.flush()	

#     # receive lines by simulator until EOM ('\n\n')
#     while (True):
#         message = simulator.stdout.readline()

#         # report message
#         sys.stderr.write('game-coordinator RECEIVED: ' + message)
#         sys.stderr.flush()

#         if (message == '\n'):
#             break


# terminate 
simulator.terminate()
simulator.stdin.close()



