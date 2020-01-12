

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

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

f_activation = nn.LeakyReLU()
# f_activation = nn.ReLU()

print(f'nonlinearity = {f_activation}')

# embeddings
MAX_TOK_POKEMON      = 893
MAX_TOK_TYPE         = 20
MAX_TOK_MOVE         = 1003
MAX_TOK_MOVE_TYPE    = 20
MAX_TOK_ABILITY      = 261
MAX_TOK_ITEM         = 1257
MAX_TOK_CONDITION    = 8
MAX_TOK_WEATHER      = 9
MAX_TOK_FIELD        = 2
MAX_TOK_ALIVE        = 2
MAX_TOK_DISABLED     = 2
MAX_TOK_SPIKES       = 4
MAX_TOK_TOXSPIKES    = 3

# imputations
VAL_STAT =     torch.tensor([2/8, 2/7, 2/6, 2/5, 2/4, 2/3, 2/2, 3/2, 4/2, 5/2, 6/2, 7/2, 8/2])
VAL_ACCURACY = torch.tensor([3/9, 3/8, 3/7, 3/6, 3/5, 3/4, 3/3, 4/3, 5/3, 6/3, 7/3, 8/3, 9/3])
VAL_EVASION =  torch.flip(VAL_ACCURACY, dims=[0])
VAL_OFFSET =   torch.tensor([6], dtype=torch.long)

FIELD_EFFECTS = [ # all true/false
    'trickroom',
    'tailwind',
    'tailwindopp',
    'encore',
    'encoreopp',
    'seed',
    'seedopp',
    'sub',
    'subopp',
    'taunt',
    'tauntopp',
    'torment',
    'tormentopp',
    'twoturnmove',
    'twoturnmoveopp',
    'confusion',
    'confusionopp',
    'stealthrock',
    'stealthrockopp',
    'reflect',
    'reflectopp',
    'lightscreen',
    'lightscreenopp'
]

class SingleEmbedding1(torch.nn.Module):
    def __init__(self, setting):
        super(SingleEmbedding1, self).__init__()
        self.embedding = torch.nn.Embedding(
            setting['dict_size'], 
            setting['embed_dim']) 
        self.dim = setting['embed_dim']

    def forward(self, pokemon_state):    
        x = self.embedding(pokemon_state)
        return x

class State1(torch.nn.Module):
    '''Returns original dict with all tokens replaced by its embeddings'''
    def __init__(self, settings):
      
        super(State1, self).__init__()

        # `key`s in dict associated with desired embedding
        self.type_fields = set(['pokemontype1', 'pokemontype2', 'movetype'])
        self.move_fields = set(['moveid'])
        self.condition_fields = set(['condition'])
   
        # manual impuations
        self.opponent_field = set(['opponent'])

        # main embeddings
        self.type_embedding = SingleEmbedding1(settings['type'])
        self.move_embedding = SingleEmbedding1(settings['move'])
        self.condition_embedding = SingleEmbedding1(settings['condition'])

        # random initialization
        # for _, param in self.named_parameters():
        #     # param.data.normal_(mean=0.0, std=0.001) 
        #     param.data.uniform_(-1.0, 1.0) 

    def __recursive_replace(self, x):

        '''Recursively replaces tokens with embeddings in dict x'''
        for key, value in x.items():

            # special case: opponent subdict (manual stat imputation in addition to recursive replace)
            if key in self.opponent_field:
                # also embeddings in subdict
                x[key] = self.__recursive_replace(value)
                # impute boosted stats for opponent
                for a, b in zip(['atk', 'def','spa','spd','spe'], 
                                ['oppatk', 'oppdef','oppspa','oppspd','oppspe']):
                    x[key]['active']['stats'][a] *= VAL_STAT[VAL_OFFSET + x[key]['boosts'][b].long().cpu()].to(DEVICE)
                           
            # if dict, not at leaf yet, so recursively replace
            elif isinstance(value, dict):
                x[key] = self.__recursive_replace(value)
            
            # if int, we possibly need an embedding
            elif isinstance(value, np.ndarray):

                # old torch compatibility
                if isinstance(value[0], np.bool_):
                    value = np.array(value, dtype=np.uint8)

                # check for embeddings
                if key in self.type_fields:
                    x[key] = self.type_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.move_fields:
                    x[key] = self.move_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.condition_fields:
                    x[key] = self.condition_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
               
                # regular value, no embedding necessary
                else:
                    # check if its of type bool and if so map to a uint8. know not empty so check first element
                    if isinstance(value[0], np.bool_):
                        value = np.array(value, dtype=np.uint8)
                    x[key] = torch.tensor(value, dtype=torch.float, device=DEVICE).unsqueeze(1)

                # print(key, x[key].shape, key) # shape debugging

            # if neither int nor dict, invalid state dict formatting
            else:
                raise ValueError('Every value in state dict must contain either integer or dict. Found: {}'.format(value))
                            
        return x

    def forward(self, x): 
        x = self.__recursive_replace(x)
        return x

# style and implementations inspired by https://nlp.seas.harvard.edu/2018/04/03/attention.html 

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def __attention(self, query, key, value, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)        
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)        
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        
    def forward(self, query, key, value):
        # 0) shape: batch_size * seq_len * d_model
        assert(key.dim() == 3 and query.dim() == 3 and value.dim() == 3)
        batch_size = query.size(0)

        # 1) Perform all the linear operations (in batch from d_model -> h * d_k) and split into h heads
        query = self.W_q(query).view(batch_size, -1, self.h, self.d_k)
        key   = self.W_k(key).view(batch_size, -1, self.h, self.d_k)
        value = self.W_v(value).view(batch_size, -1, self.h, self.d_k)

        # 2) Transpose to get dimensions batch_size * heads * seq_len * d_k
        query = query.transpose(1,2)
        key   = key.transpose(1,2)
        value = value.transpose(1,2)

        # 3) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.__attention(query, key, value, dropout=self.dropout)
        
        # # 4) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.h * self.d_k)
        x = self.W_o(x)
        return x


class SelfAttention1(nn.Module):
    '''
    Self attention
    Permutation EQUIvariant
    '''

    def __init__(self, heads, d_model, dropout=0.0):
        super(SelfAttention1, self).__init__()
        self.size = d_model
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.sublayer = lambda x: self.attn(x, x, x)
       
    def forward(self, x):
        assert(x.dim() == 3) # batchsize, setsize, features
        "Apply residual connection to any sublayer with the _same size_."
        return self.sublayer(x)

class DeepSet1(nn.Module):
    '''
    Simple Permutation Invariant Module
    https://arxiv.org/pdf/1703.06114.pdf 
    '''
    def __init__(self, phi, rho):
        super(DeepSet1, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        assert(x.dim() == 3) # batch, setsize, d_in
        return self.rho(self.phi(x).sum(dim=1))


'''
Main representation branches
'''

class MoveRepresentation1(nn.Module):
    '''
    Creates representation of `move_state` 
    '''
    def __init__(self, s, dropout=0.0):
        super(MoveRepresentation1, self).__init__()
        
        self.d_move = s['move']['embed_dim']
        self.d_move_type = s['type']['embed_dim']

        self.cat_dim = self.d_move + self.d_move_type 
        self.d_out = self.d_move  # define this to be out dim

        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.cat_dim), f_activation,
            nn.Linear(self.cat_dim, self.cat_dim), f_activation,
            nn.Linear(self.cat_dim, self.d_out),
        )

    def forward(self, x):
        h = [
             x['moveid'],
             x['movetype'],
        ]

        h = torch.cat(h, dim=1)
        assert(h.shape[1] == self.cat_dim)

        # final
        h = self.final(h)
        return h

class PokemonRepresentation1(nn.Module):
    '''
    Creates representation of `pokemon_state` 
    '''
    def __init__(self, s, d_pokemon, dropout=0.0, attention=False):
        super(PokemonRepresentation1, self).__init__()

        self.d_type = s['type']['embed_dim']
        self.d_condition = s['condition']['embed_dim']

        # shared move representation learned for each move (permutation EQUIvariance)  
        self.move_embed = MoveRepresentation1(s, dropout=dropout)
        self.d_move = self.move_embed.d_out # deep set move representation

        self.move_relate = SelfAttention1(heads=4, d_model=self.d_move, dropout=dropout) if attention else nn.Identity()

        self.cat_dim = self.d_move # collective move representation
        self.cat_dim += 2 * self.d_type + self.d_condition  # pokemon info
        self.cat_dim += 6 # ints in pokemon state

        self.d_out = d_pokemon # define this to be out dimension

        # relationship deep set function (permutation INvariance)      
        self.move_DS = DeepSet1(
            nn.Sequential( # phi
                nn.Linear(self.d_move, self.d_move), f_activation, 
                nn.Linear(self.d_move, self.d_move)),
            nn.Sequential(  # rho
                nn.Linear(self.d_move, self.d_move), f_activation, 
                nn.Linear(self.d_move, self.d_move)))
        
        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.cat_dim), f_activation,
            nn.Linear(self.cat_dim, self.cat_dim), f_activation,
            nn.Linear(self.cat_dim, self.d_out),
        )


    def forward(self, pokemon):

        # DEBUG
        # print('input')
        # for i in range(4):
        #     pprint(x['moves'][i])
        # print('\n output')
        # pprint(moves)

        x = pokemon
        
        # order equivariant move representations
        moves = torch.stack([self.move_embed(x['moves'][i]) for i in range(4)], dim=1) # bs, moves, d
        
        moves_equivariant = self.move_relate(moves)
        
        # order invariant deep sets representation of all moves
        moves_invariant = self.move_DS(moves_equivariant)

        # pokemon representation
        if (x['stats']['max_hp'] == 0).sum() != 0:
            relative_hp = x['hp'] * 0.0
        else:
            relative_hp = x['hp'] / x['stats']['max_hp']

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
        h = self.final(h)

        # return (hidden representation, equivariant move representations (for policy))
        return h, moves_equivariant

class PlayerRepresentation1(nn.Module):
    '''
    `player`/  and `opponent/` 
    '''
    def __init__(self, d_out, s, d_pokemon, dropout=0.0, attention=False):
        super(PlayerRepresentation1, self).__init__()

        # pokemon representations (shared/permutation equivariant for team pokemon)
        self.d_pokemon = d_pokemon
        self.active_pokemon = PokemonRepresentation1(s, d_pokemon=self.d_pokemon, dropout=dropout, attention=attention)
        self.team_pokemon = PokemonRepresentation1(s, d_pokemon=self.d_pokemon, dropout=dropout, attention=attention)

        # dims
        assert(self.active_pokemon.d_out == self.team_pokemon.d_out 
            and self.active_pokemon.d_move == self.team_pokemon.d_move)
        self.d_move = self.active_pokemon.d_move 
        self.d_out = d_out
        self.cat_dim = 2 * self.d_pokemon

        self.team_pokemon_relate = SelfAttention1(
            heads=4, d_model=self.d_pokemon, dropout=dropout) if attention else nn.Identity()


        # team pokemon relationship deep set function (permutation INvariance)    
        self.team_DS = DeepSet1(
            nn.Sequential( # phi
                nn.Linear(self.d_pokemon, self.d_pokemon), f_activation, 
                nn.Linear(self.d_pokemon, self.d_pokemon)),
            nn.Sequential(  # rho
                nn.Linear(self.d_pokemon, self.d_pokemon), f_activation,
                nn.Linear(self.d_pokemon, self.d_pokemon)))
            
        # final         
        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.cat_dim), f_activation,
            nn.Linear(self.cat_dim, self.cat_dim), f_activation,
            nn.Linear(self.cat_dim, self.d_out),
        )

        
        
    def forward(self, x):

        # active pokemon representation
        # (invariant, equivariant)
        active_pokemon, moves_equivariant = self.active_pokemon(x['active'])

        # team pokemon representation (equivariant move reps are discarded for individual pokemon reps)
        team_pokemon = torch.stack([self.team_pokemon(x['team'][i])[0] for i in range(6)], dim=1)
        team_pokemon_equivariant = self.team_pokemon_relate(team_pokemon)

        # invariant
        team = self.team_DS(team_pokemon_equivariant)

        # combine into player representation
        player = self.final(torch.cat([active_pokemon, team], dim=1))

        # return (player, moves_equivariant, team_pokemon_equivariant)
        return player, moves_equivariant, team_pokemon_equivariant



'''
Final net
'''

class SmallDeePikachu(nn.Module):
    '''Value and Policy Net'''

    def __init__(self, state_embedding_settings, hidden_layer_settings, dropout=0.0, attention=False):
        super(SmallDeePikachu, self).__init__()

        self.hidden_layer_settings = hidden_layer_settings
        self.d_player = hidden_layer_settings['player']
        self.d_opp = hidden_layer_settings['opponent']
        self.d_context = hidden_layer_settings['context']
        self.d_pokemon = hidden_layer_settings['pokemon_hidden']

        self.d_hidden_in = self.d_player + self.d_opp
        
        self.state_embedding = State1(state_embedding_settings)
        self.state_embedding_settings = state_embedding_settings

        # major hidden state 
        self.player = PlayerRepresentation1(d_out=self.d_player, s=state_embedding_settings, d_pokemon=self.d_pokemon, attention=attention, dropout=dropout)
        self.opponent = PlayerRepresentation1(d_out=self.d_opp, s=state_embedding_settings, d_pokemon=self.d_pokemon, attention=attention, dropout=dropout)
        self.d_move = self.player.d_move

        self.hidden_reduce = nn.Sequential(
            nn.Linear(self.d_hidden_in, self.d_hidden_in), f_activation,
            nn.Linear(self.d_hidden_in, self.d_context), f_activation,
            nn.Linear(self.d_context, self.d_context),
        )

        # value function
        self.value_function = nn.Sequential(
                nn.Linear(self.d_context, self.d_context), f_activation,
                nn.Linear(self.d_context, 1), 
            )
        
        # Q function (heads 1 and 2) 
        self.q_combine_moves_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_move + self.d_context, self.d_context), f_activation,
                nn.Linear(self.d_context, 1), 
            ),
            nn.Sequential(
                nn.Linear(self.d_move + self.d_context, self.d_context), f_activation,
                nn.Linear(self.d_context, 1),  
            )])
        
        self.q_combine_pokemon_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_pokemon + self.d_context, self.d_context), f_activation,
                nn.Linear(self.d_context, 1), 
            ),
            nn.Sequential(
                nn.Linear(self.d_pokemon + self.d_context,
                          self.d_context), f_activation,
                nn.Linear(self.d_context, 1), 
            )])
        
        
    def forward(self, x):
        
        state = copy.deepcopy(x) # embedding is put inplace

        # # DEBUG
        # print('---- STATE')
        # for j in range(4):
        #     pprint(state['player']['active']['moves'][j])

        state = self.state_embedding(state)

        # player 
        player, moves_equivariant, team_pokemon_equivariant = self.player(state['player'])

        # opponent
        opponent, _, _ = self.opponent(state['opponent'])

        # combine hidden representations of player, opponent into context
        hidden = torch.cat([player, opponent], dim=1)
        
        context = self.hidden_reduce(hidden)


        # value function
        value = self.value_function(context).squeeze(dim=1)

        # q function
        moves_and_context = torch.cat(
            [moves_equivariant, 
             context.unsqueeze(1).repeat((1, 4, 1))], 
        dim=2)

        pokemon_and_context = torch.cat(
            [team_pokemon_equivariant, 
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

    example_state = state.create_2D_state(2)

    state_embedding_settings = {
        'move':        {'embed_dim': 8, 'dict_size': MAX_TOK_MOVE},
        'type':        {'embed_dim': 4, 'dict_size': MAX_TOK_TYPE},
        'condition':   {'embed_dim': 4, 'dict_size': MAX_TOK_CONDITION},
    }

    hidden_layer_settings = {
        'pokemon_hidden' : 16,
        'player' : 16,
        'opponent' : 16,
        'context' : 16,
    }

    model = SmallDeePikachu(
        state_embedding_settings,
        hidden_layer_settings,
        dropout=0.0,
        attention=True)

    
    out = model(copy.deepcopy(example_state))

    print(out[1])



