
### Alakazam

Alakazam is a reinforcement learning agent that learns to play the generation 5 Overused format of competitive Pokemon on Pokemon Showdown. It was part of a study on equivariant and invariant neural networks for use in decision making applications. 

------------------------------------------------------------------------

The code make use of the Pokemon Showdown game engine. Therefore the code builds on the server code for Pokemon Showdown [1]. We list here the code written specifically for this project. Any files not included in this list and not in the original Pokemon Showdown server code are deprecated files. 

### training_parallel.py

The main code that should be run to fully train an agent against an opponent that plays randomly using the teams in construct_teams/team_folder.

### neural_net_small_2.py

Defines the architecture for the neural network used in our experiments. 

### neural_net_small_2_baseline.py

Defines the architecture for the neural network used as a baseline in our experiments. 

### game_coordinator.py

Implements framework to simulate a game using ./pokemon-showdown simulate-battle using user defined agents.
When executed, simulates a game using the two provided agents. If no agents provided, uses default players.

Define agent types used as follows (if `-p1 agent` or `-p2 agent` not passed, `default` agent is used)

**\# python game_coordinator.py -p1 default -p2 default**

### game_coordinator_parallel.py

The same as the above file but used to run multiple games at once. This makes training much faster since one forward pass of the neural network handles actions in multiple concurrent games. 

Also contains an experience replay class for storing experience from all distributed agents. This file also contains the ParallelLearningAgent which is the agent class used for training, inheriting from the classes in training.py. Overrides the SACAgent to implement Q learning. 

### agents.py

Defines agent behavior. Includes functions to construct a valid game state from game engine messages and establish valid actions. 

### custom_structures.py

Defines structures reoccuring in simulations, such as messages by the simulator or actions by the agents.

### state.py

Defines the default state space. 

### training.py

Contains a heirachy of classes used to define an agent that can store previous experience. 

### data_pokemon.py 

Defines mappings from game information to unique integers.

### teams_data.py

Inc;lues some sample teams and a function that generates teams of a given size using the teams in construct_teams/team_folder. 

### construct_teams

A folder that was used to construct teams from the file gen5ou.txt which was obtained using [2]. construct_teams/team_gen.js is first run on this file to construct a json object representation (in construct_teams/team_folder), before running team_export_to_line.py which converts the teams to simulator-readable format. These teams are kept in construct_teams/team_gen.js. 


  [1]: https://github.com/smogon/pokemon-showdown
  [2]: https://www.smogon.com/forums/threads/smogon-rmt-team-dump.3622884/

Installing
------------------------------------------------------------------------

For the Pokemon Showdown server code:

    ./pokemon-showdown

(Requires Node.js v10+)

If your distro package manager has an old Node.js version, the simplest way to upgrade is `n` – usually no root necessary:

    npm install --global n
    n latest

To run our code you can create a conda environment with the required specifications by running

    conda create --name myenv --file spec-file.txt

See cluster_instructions.txt for information to run on Leonhard. 

To run the team generation materials you need to install koffing with "npm I --save koffing". https://github.com/route1rodent/koffing

Running Code
------------------------------------------------------------------------

An agent can be trained by running 

    python3 training_parallel.py
