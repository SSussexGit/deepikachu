

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from pprint import pprint
import math
import state
import time
import json

from neural_net_small_2 import *

class BaselinePokemonRepresentation2(PokemonRepresentation2):

    def __init__(self, s, d_pokemon, move_identity, layer_norm, dropout=0.0, attention=False):
        super(BaselinePokemonRepresentation2, self).__init__(
            s, d_pokemon, move_identity, layer_norm, dropout=dropout, attention=attention)

        self.d_type = s['type']['embed_dim']
        self.d_condition = s['condition']['embed_dim']

        # shared move representation learned for each move (permutation EQUIvariance)
        self.move_representation = MoveRepresentation2(
            s, move_identity=move_identity, layer_norm=layer_norm, dropout=dropout)

        # self.d_move = 2 * self.move_representation.d_out
        self.d_move = self.move_representation.d_out  # no self attention

        # self.move_relate = SelfAttention2(
        #     heads=4, d_model=self.move_representation.d_out, dropout=dropout) if attention else make_identity()
        self.move_relate = None # no self attention


        self.cat_dim = self.d_move  # collective move representation
        self.cat_dim += 2 * self.d_type + self.d_condition  # pokemon info
        self.cat_dim += 6  # ints in pokemon state

        # self.d_out = d_pokemon  # define this to be out dimension

        # feedforward instead of deep set
        self.move_DS = nn.Sequential( 
            nn.Linear(4 * self.d_move,  4 * self.d_move), make_f_activation(),
            nn.Linear(4 * self.d_move,  4 * self.d_move), make_f_activation(),
            nn.Linear(4 * self.d_move, self.d_move))
                

        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.cat_dim), make_f_activation(),
            nn.Linear(self.cat_dim, self.d_out), nn.LayerNorm(
                self.d_out) if layer_norm else make_identity(),
        )

    def forward(self, pokemon):

        x = pokemon

        # order equivariant move representations
        moves = []
        moves_types = []
        for i in range(4):
            move_rep, move_type = self.move_representation(x['moves'][i])
            moves.append(move_rep)
            moves_types.append(move_type)
        moves = torch.stack(moves, dim=1)  # bs, moves, d
        moves_types_equivariant = torch.stack(
            moves_types, dim=1)  # bs, moves, d

        # cat representation with attention representation
        # moves_equivariant = torch.cat(
        #     [moves, self.move_relate(moves)], dim=2)

        moves_equivariant = moves # no self attention
   
        # order invariant deep sets representation of all moves
        moves_invariant = self.move_DS(moves_equivariant.reshape(moves_equivariant.shape[0], -1))

        # pokemon representation
        max_hp = copy.deepcopy(x['stats']['max_hp'])
        # so no div by 0 if maxhp = 0
        max_hp = max_hp.masked_fill(max_hp == 0, 1.0).float()
        relative_hp = x['hp'] / max_hp * HP_SCALE

        h = [
            moves_invariant,
            x['condition'],
            x['pokemontype1'],
            x['pokemontype2'],
            x['stats']['atk'],
            x['stats']['def'],
            x['stats']['spa'],
            x['stats']['spd'],
            x['stats']['spe'],
            relative_hp,
        ]

        h = torch.cat(h, dim=1)
        assert(h.shape[1] == self.cat_dim)

        # final
        pokemon_invariant = self.final(h)

        pokemon_types = (x['pokemontype1'], x['pokemontype2'])
        pokemon_stats = torch.cat([
            x['stats']['atk'],
            x['stats']['def'],
            x['stats']['spa'],
            x['stats']['spd'],
            x['stats']['spe'],
            relative_hp], dim=1)

        # return (hidden representation, equivariant move representations (for policy))
        return (pokemon_invariant,
                moves_equivariant,
                moves_types_equivariant,
                pokemon_stats,
                pokemon_types,
                relative_hp)
            

class BaselinePlayerRepresentation2(PlayerRepresentation2):
    '''
    `player`/  and `opponent/` 
    '''

    def __init__(self, d_out, s, d_pokemon, move_identity, layer_norm, dropout=0.0, attention=False):
        super(BaselinePlayerRepresentation2, self).__init__(
            d_out, s, d_pokemon, move_identity, layer_norm, dropout=dropout, attention=attention)

        # pokemon representations (shared/permutation equivariant for team pokemon)
        # self.d_pokemon = 2 * d_pokemon
        self.d_pokemon = d_pokemon # no self attention
        self.active_pokemon = BaselinePokemonRepresentation2(
            s, d_pokemon=d_pokemon, move_identity=move_identity, dropout=dropout, attention=attention, layer_norm=layer_norm)
        self.team_pokemon = BaselinePokemonRepresentation2(
            s, d_pokemon=d_pokemon, move_identity=move_identity, dropout=dropout, attention=attention, layer_norm=layer_norm)

        # dims
        assert(self.active_pokemon.d_out == self.team_pokemon.d_out
               and self.active_pokemon.d_move == self.team_pokemon.d_move)
        self.d_move = self.active_pokemon.d_move
        self.d_out = d_out
        # self.cat_dim = d_pokemon + 2 * d_pokemon  # active + team
        self.cat_dim = 2 * d_pokemon  # no self attention

        # self.team_pokemon_relate = SelfAttention2(
        #     heads=4, d_model=d_pokemon, dropout=dropout) if attention else make_identity()

        self.team_pokemon_relate = None  # no self attention

        # team pokemon relationship deep set function (permutation INvariance)
        # self.team_DS = DeepSet2(
        #     nn.Sequential(  # phi
        #         nn.Linear(self.d_pokemon, self.d_pokemon), make_f_activation(),
        #         nn.Linear(self.d_pokemon, self.d_pokemon)),
        #     nn.Sequential(  # rho
        #         nn.Linear(self.d_pokemon, self.d_pokemon), make_f_activation(),
        #         nn.Linear(self.d_pokemon, self.d_pokemon)))

        # feedforward instead of deep set
        self.team_DS = nn.Sequential(
            nn.Linear(6 * self.d_pokemon,  6 * self.d_pokemon), make_f_activation(),
            nn.Linear(6 * self.d_pokemon,  6 * self.d_pokemon), make_f_activation(),
            nn.Linear(6 * self.d_pokemon, self.d_pokemon))


        # final
        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.cat_dim), make_f_activation(),
            nn.Linear(self.cat_dim, self.d_out), nn.LayerNorm(
                self.d_out) if layer_norm else make_identity()
        )

    def forward(self, x):

        # active pokemon representation
        (active_invariant,
            active_moves_equivariant,
            active_moves_types_equivariant,
            active_stats,
            active_types,
            active_hp) = \
            self.active_pokemon(x['active'])

        # team pokemon representation
        team_pokemon = []
        team_pokemon_types1 = []
        team_pokemon_types2 = []
        team_pokemon_hp = []
        for i in range(6):
            poke_rep, _, _, _, poke_types, poke_hp = self.team_pokemon(
                x['team'][i])
            team_pokemon.append(poke_rep)
            team_pokemon_types1.append(poke_types[0])
            team_pokemon_types2.append(poke_types[1])
            team_pokemon_hp.append(poke_hp)

        team_pokemon = torch.stack(team_pokemon, dim=1)  # bs, pokemon, d
        team_pokemon_types1 = torch.stack(
            team_pokemon_types1, dim=1)  # bs, pokemon, d
        team_pokemon_types2 = torch.stack(
            team_pokemon_types2, dim=1)  # bs, pokemon, d
        team_pokemon_hp = torch.stack(team_pokemon_hp, dim=1)  # bs, pokemon, d

        # get alive status for masking in attention and deep set
        self.team_alive = torch.cat(
            [x['team'][i]['alive'] for i in range(6)], dim=1)
        self.team_alive = self.team_alive.long().to(DEVICE)

        # cat representation with attention representation

        # no self attentions
        # team_pokemon_equivariant = torch.cat(
        #     [team_pokemon,  self.team_pokemon_relate(team_pokemon, self.team_alive)], dim=2)
        
        team_pokemon_equivariant = team_pokemon
        
  
        # invariant
        # team = self.team_DS(team_pokemon_equivariant, self.team_alive)
        team = self.team_DS(team_pokemon_equivariant.reshape(team_pokemon_equivariant.shape[0], -1))

        # combine into player representation
        player = self.final(torch.cat([active_invariant, team], dim=1))

        return (player,
                active_moves_equivariant,
                active_moves_types_equivariant,
                team_pokemon_equivariant,
                active_stats,
                active_types,
                (team_pokemon_types1, team_pokemon_types2),
                active_hp,
                team_pokemon_hp)



class BaselineSmallDeePikachu2(SmallDeePikachu2):
    '''Value and Policy Net'''

    def __init__(self, state_embedding_settings, hidden_layer_settings, layer_norm=False, move_identity=False, dropout=0.0, attention=False):
        super(BaselineSmallDeePikachu2, self).__init__(state_embedding_settings,
            hidden_layer_settings, layer_norm=layer_norm, move_identity=move_identity, dropout=dropout, attention=attention)

        self.f_activation = f_activation
        self.move_identity = move_identity
        self.layer_norm = layer_norm

        self.hidden_layer_settings = hidden_layer_settings
        self.d_player = hidden_layer_settings['player']
        self.d_opp = hidden_layer_settings['opponent']
        self.d_context = hidden_layer_settings['context']

        # similarity metrics
        self.d_type = state_embedding_settings['type']['embed_dim']

        # this will result in h * h similarity metrics per type
        self.sim_heads_opp = 3 
        self.sim_heads_p = 2

        # self.d_sim = self.d_type // 2 # arbitrary
        # self.d_sim_opp_out = self.sim_heads_opp * self.sim_heads_opp
        # self.d_sim_p_out = self.sim_heads_p * self.sim_heads_p

        # (player) + (opponent) + (similarity of four active type pairs) + (stats of both players active) + (hp summaries team)
        # self.d_hidden_in = self.d_player + self.d_opp + \
        #     4 * self.d_sim_opp_out + 12 + 2
        
        self.d_hidden_in = self.d_player + self.d_opp + 12 + 2 # no similarity

        self.state_embedding = State2(state_embedding_settings)
        self.state_embedding_settings = state_embedding_settings

        # major hidden state 
        self.player = BaselinePlayerRepresentation2(
            d_out=self.d_player, s=state_embedding_settings, d_pokemon=hidden_layer_settings['pokemon_hidden'], 
            move_identity=move_identity, attention=attention, dropout=dropout, layer_norm=layer_norm)

        self.opponent = BaselinePlayerRepresentation2(
            d_out=self.d_opp, s=state_embedding_settings, d_pokemon=hidden_layer_settings['pokemon_hidden'], 
            move_identity=move_identity, attention=attention, dropout=dropout, layer_norm=layer_norm)

        self.d_move = self.player.d_move
        self.d_pokemon = self.player.d_pokemon

        self.hidden_reduce = nn.Sequential(
            nn.Linear(self.d_hidden_in, self.d_hidden_in), make_f_activation(),
            nn.Linear(self.d_hidden_in, self.d_context), nn.LayerNorm(self.d_context) if layer_norm else make_identity(),
        )
       

        # # similarity metrics
        # # A = player moves - opp active
        # # B = player team  - opp active
        # # C = player moves - player active
        # self.A_p = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim) # p move - opp active
        # self.A_opp = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim) 
        
        # self.B_p = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim) # p team - opp active
        # self.B_opp = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim)

        # self.C_p_move = nn.Linear(self.d_type, self.sim_heads_p * self.d_sim) # p move - p active
        # self.C_p_active = nn.Linear(self.d_type, self.sim_heads_p * self.d_sim)

        # value function
        self.value_function = nn.Sequential(
                nn.Linear(self.d_context, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1), 
            )
        
        # Q function (heads 1 and 2) 
        # self.d_q_move_in = self.d_move + self.d_context + 2 * self.d_sim_opp_out + 2 * self.d_sim_p_out + 4 # hp summaries
        # self.d_q_team_in = self.d_pokemon + self.d_context + 4 * self.d_sim_opp_out + 5 # team i hp + hp summaries

        self.d_q_move_in = self.d_move + self.d_context + 4 # no sim
        self.d_q_team_in = self.d_pokemon + self.d_context + 5 # no sim

        self.q_combine_moves_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_q_move_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1),
            ),
            nn.Sequential(
                nn.Linear(self.d_q_move_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1),
            )])
        
        self.q_combine_pokemon_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_q_team_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1),
            ),
            nn.Sequential(
                nn.Linear(self.d_q_team_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1),
            )])

        # random init
        for s, param in self.named_parameters():
            if param.dim() == 1:
                # bias, this is established practice
                nn.init.zeros_(param)
            else:
                # weight
                if 'embedding' in s:
                    nn.init.xavier_uniform_(param, 
                        gain=torch.nn.init.calculate_gain('leaky_relu'))
                else:
                    nn.init.xavier_uniform_(param, 
                        gain=torch.nn.init.calculate_gain('leaky_relu'))

        
    def forward(self, x):
        
        state = copy.deepcopy(x) # embedding is put inplace
        state = self.state_embedding(state)

        # player 
        (player, 
            player_moves_equivariant,
            player_moves_types_equivariant,
            player_team_equivariant, 
            player_active_stats, 
            (player_active_type1, player_active_type2),
            (player_team_pokemon_types1, player_team_pokemon_types2),
            player_active_hp,
            player_team_hp) = \
            self.player(state['player'])

        # opponent
        (opponent, _, _, _,
            opponent_active_stats,
            (opponent_active_type1, opponent_active_type2),
            (_, _),
            opponent_active_hp,
            opponent_team_hp) = \
            self.opponent(state['opponent'])

        # hp average
        player_team_hp_ave = player_team_hp.sum(dim=1) / 6.0
        opponent_team_hp_ave = opponent_team_hp.sum(dim=1) / 6.0

        # # similarity metrics comparing (looks more than it is)
        # # player active move types - opponent active pokemon types
        # # player team pokemon types - opponent active pokemon types
        # mv_types = player_moves_types_equivariant
        # team_type1 = player_team_pokemon_types1
        # team_type2 = player_team_pokemon_types2
        # p_active_type1 = player_active_type1.unsqueeze(1)
        # p_active_type2 = player_active_type2.unsqueeze(1)

        # opp_active_type1 = opponent_active_type1.unsqueeze(1)
        # opp_active_type2 = opponent_active_type2.unsqueeze(1)

        # batch_size = mv_types.shape[0]

        # # perform all the linear operations (in batch from d_type -> h * d_similarity) and split into h heads
        # move_p = self.A_p(mv_types).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        # move_opp1 = self.A_opp(opp_active_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        # move_opp2 = self.A_opp(opp_active_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)

        # team_p1 = self.B_p(team_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        # team_p2 = self.B_p(team_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        # team_opp1 = self.B_opp(opp_active_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        # team_opp2 = self.B_opp(opp_active_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        # active_p1 = self.B_p(p_active_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim) # comp active as well 
        # active_p2 = self.B_p(p_active_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)

        # stab_move = self.C_p_move(mv_types).view(batch_size, -1, self.sim_heads_p, self.d_sim)
        # stab_active1 = self.C_p_active(p_active_type1).view(batch_size, -1, self.sim_heads_p, self.d_sim)
        # stab_active2 = self.C_p_active(p_active_type2).view(batch_size, -1, self.sim_heads_p, self.d_sim)
              
        # # # compute similarity scores, and concat using .view()
        # # A
        # mv_scores1 = self.__dot_similarity(move_p, move_opp1, self.sim_heads_opp)
        # mv_scores2 = self.__dot_similarity(move_p, move_opp2, self.sim_heads_opp)

        # # B
        # team_scores11 = self.__dot_similarity(team_p1, team_opp1, self.sim_heads_opp)
        # team_scores12 = self.__dot_similarity(team_p1, team_opp2, self.sim_heads_opp)
        # team_scores21 = self.__dot_similarity(team_p2, team_opp1, self.sim_heads_opp)
        # team_scores22 = self.__dot_similarity(team_p2, team_opp2, self.sim_heads_opp)
        
        # # type scores: p active - opp active
        # active_scores = torch.cat([
        #     self.__dot_similarity(active_p1, team_opp1, self.sim_heads_opp).squeeze(1),
        #     self.__dot_similarity(active_p1, team_opp2, self.sim_heads_opp).squeeze(1),
        #     self.__dot_similarity(active_p2, team_opp1, self.sim_heads_opp).squeeze(1),
        #     self.__dot_similarity(active_p2, team_opp2, self.sim_heads_opp).squeeze(1),
        # ], dim=1) 
        
        # # C
        # mv_scores_own1 = self.__dot_similarity(stab_move, stab_active1, self.sim_heads_p)
        # mv_scores_own2 = self.__dot_similarity(stab_move, stab_active2, self.sim_heads_p)


        # combine hidden representations of 
        # player, opponent, active sims, hps into context 
        hidden = torch.cat([
            player, 
            player_active_stats, 
            # active_scores, 
            opponent,
            opponent_active_stats,
            player_team_hp_ave,
            opponent_team_hp_ave,
        ], dim=1)

        context = self.hidden_reduce(hidden)

        # value function
        value = self.value_function(context).squeeze(dim=1)

        # q function (add similarity scores of types)
        moves_and_context = torch.cat(
            [player_moves_equivariant,
            #  mv_scores1, mv_scores2,  # 2 * 4 metrics of similarity (comp w 2x opp type)
            #  mv_scores_own1, mv_scores_own2, # 2 * 4 metrics of similarity(comp w 2x own type),
             player_active_hp.unsqueeze(1).repeat((1, 4, 1)), # own hp
             player_team_hp_ave.unsqueeze(1).repeat((1, 4, 1)),
             opponent_active_hp.unsqueeze(1).repeat((1, 4, 1)), # opp hp
             opponent_team_hp_ave.unsqueeze(1).repeat((1, 4, 1)),
             context.unsqueeze(1).repeat((1, 4, 1))
             ], 
        dim=2)

        pokemon_and_context = torch.cat(
            [player_team_equivariant,
            #  team_scores11, team_scores12, team_scores21, team_scores22, # 4 * 4 metrics of similarity (comp 2x own type w 2x opp type)
             player_team_hp, # team hp individual (equivariant)
             player_active_hp.unsqueeze(1).repeat((1, 6, 1)), # own hp
             player_team_hp_ave.unsqueeze(1).repeat((1, 6, 1)),
             opponent_active_hp.unsqueeze(1).repeat((1, 6, 1)), # opp hp
             opponent_team_hp_ave.unsqueeze(1).repeat((1, 6, 1)),
             context.unsqueeze(1).repeat((1, 6, 1))],
        dim=2)

        all_actions_A =  torch.cat([
            self.q_combine_moves_context[0](moves_and_context), 
            self.q_combine_pokemon_context[0](pokemon_and_context)], dim=1)
        q_values_A = all_actions_A.squeeze(dim=2)

        all_actions_B =  torch.cat([
            self.q_combine_moves_context[1](moves_and_context), 
            self.q_combine_pokemon_context[1](pokemon_and_context)], dim=1)
        q_values_B = all_actions_B.squeeze(dim=2)

        return q_values_A, q_values_B, value


if __name__ == '__main__':

    '''
    Example
    '''
    state_embedding_settings = {
        'move':        {'embed_dim': 32, 'dict_size': MAX_TOK_MOVE},
        'type':        {'embed_dim': 16, 'dict_size': MAX_TOK_TYPE},
        'condition':   {'embed_dim': 8, 'dict_size': MAX_TOK_CONDITION},
        'move_category':   {'embed_dim': 8, 'dict_size': MAX_TOK_MOVE_CATEGORY},
    }

    hidden_layer_settings = {
        'player' : 64,
        'opponent' : 64,
        'context' : 64,
        'pokemon_hidden' : 32,

    }

    # player 1 neural net (initialize target network the same)
    p1net = SmallDeePikachu2(
        state_embedding_settings,
        hidden_layer_settings,
        move_identity=True,
        layer_norm=True,
        dropout=0.0,
        attention=True)

    p1net.train()

    example_state = state.create_2D_state(2)
    out = p1net(copy.deepcopy(example_state))

    print('done.')


    # baseline
    base = BaselineSmallDeePikachu2(
        state_embedding_settings,
        hidden_layer_settings,
        move_identity=True,
        layer_norm=True,
        dropout=0.0,
        attention=True)

    base.train()

    example_state = state.create_2D_state(2)
    out = base(copy.deepcopy(example_state))

    print('done.')

    
