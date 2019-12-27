

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from pprint import pprint
import math
import state
import time

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

class SingleEmbedding(torch.nn.Module):
    def __init__(self, setting):
        super(SingleEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(
            setting['dict_size'], 
            setting['embed_dim'], 
            max_norm=1.0, norm_type=2.0) 
        self.dim = setting['embed_dim']

    def forward(self, pokemon_state):    
        x = self.embedding(pokemon_state)
        return x


class State(torch.nn.Module):
    '''Returns original dict with all tokens replaced by its embeddings'''
    def __init__(self, settings):
      
        super(State, self).__init__()

        # `key`s in dict associated with desired embedding
        self.pokemon_fields = set(['pokemon_id'])
        self.type_fields = set(['pokemontype1', 'pokemontype2'])
        self.move_fields = set(['moveid', 'twoturnmoveid', 'twoturnmoveoppid'])
        self.move_type_fields = set(['movetype'])
        self.ability_fields = set(['baseAbility'])
        self.item_fields = set(['item'])
        self.condition_fields = set(['condition'])
        self.weather_fields = set(['weather'])
        self.alive_fields = set(['alive'])
        self.disabled_fields = set(['disabled'])
        self.spikes_fields = set(['spikes', 'spikesopp'])
        self.toxicspikes_fields = set(['toxicspikes', 'toxicspikesopp'])

        # manual impuations
        self.accuracy_fields = set(['accuracy'])
        self.evasion_fields = set(['evasion'])
        self.opponent_field = set(['opponent'])

        # field effect handled differently than above main embeddings
        self.fieldeffect_fields = set(FIELD_EFFECTS)

        # dict: fieldeffect string -> idx (to find correct Embedding in list)
        self.field_idx_dict = {s:j for j, s in enumerate(self.fieldeffect_fields)}

        # main embeddings
        self.pokemon_embedding = SingleEmbedding(settings['pokemon'])
        self.type_embedding = SingleEmbedding(settings['type'])
        self.move_embedding = SingleEmbedding(settings['move'])
        self.move_type_embedding = SingleEmbedding(settings['move_type'])
        self.ability_embedding = SingleEmbedding(settings['ability'])
        self.item_embedding = SingleEmbedding(settings['item'])
        self.condition_embedding = SingleEmbedding(settings['condition'])
        self.weather_embedding = SingleEmbedding(settings['weather'])
        self.alive_embedding = SingleEmbedding(settings['alive'])
        self.disabled_embedding = SingleEmbedding(settings['disabled'])
        self.spikes_embedding = SingleEmbedding(settings['spikes'])
        self.spikesopp_embedding = SingleEmbedding(settings['spikes'])
        self.toxicspikes_embedding = SingleEmbedding(settings['toxicspikes'])
        self.toxicspikesopp_embedding = SingleEmbedding(settings['toxicspikes'])

        # list of embeddings for each field effect
        self.fieldeffect_embeddings = nn.ModuleList([
            SingleEmbedding(settings['fieldeffect']) for _, s in self.field_idx_dict.items()
        ])

        # random initialization
        for _, param in self.named_parameters():
            # param.data.normal_(mean=0.0, std=0.001) 
            param.data.uniform_(-1.0, 1.0) 


    # def __convert_bool(self, b):
    #     return torch.tensor(1) if b else torch.tensor(0) 

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
                #old torch compatabilitry
                if isinstance(value[0], np.bool_):
                    value = np.array(value, dtype=np.uint8)

                if key in self.pokemon_fields:
                    x[key] = self.pokemon_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.type_fields:
                    x[key] = self.type_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.move_fields:
                    x[key] = self.move_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.move_type_fields:
                    x[key] = self.move_type_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.ability_fields:
                    x[key] = self.ability_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.item_fields:
                    x[key] = self.item_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.condition_fields:
                    x[key] = self.condition_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.weather_fields:
                    x[key] = self.weather_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.alive_fields:
                    x[key] = self.alive_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.disabled_fields:
                    x[key] = self.disabled_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.spikes_fields:
                    if key == 'spikes':
                        x[key] = self.spikes_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                    else:
                        x[key] = self.spikesopp_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                elif key in self.toxicspikes_fields:
                    if key == 'toxicspikes':
                        x[key] = self.toxicspikes_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                    else:
                        x[key] = self.toxicspikesopp_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
                # special: need to find correct embedding in list
                elif key in self.fieldeffect_fields:
                    x[key] = self.fieldeffect_embeddings[self.field_idx_dict[key]](torch.tensor(value, dtype=torch.long, device=DEVICE))

                # manual imputations
                elif key in self.accuracy_fields:
                    x[key] = (VAL_ACCURACY[VAL_OFFSET + torch.tensor(value)]).unsqueeze(1).float().to(DEVICE)
                elif key in self.evasion_fields:
                    x[key] = (VAL_EVASION[VAL_OFFSET + torch.tensor(value)]).unsqueeze(1).float().to(DEVICE)
                    

                # regular value, no embedding necessary
                else:
                    #check if its of type bool and if so map to a uint8. know not empty so check first element
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
    def __init__(self, h, d_model, dropout=0.1):
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

class FeedForward0(nn.Module):
    '''Simple FeedForward Net'''
    def __init__(self, d_in, d_ff, d_out, dropout=0.1):
        super(FeedForward0, self).__init__()
        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ResidualFeedForward0(nn.Module):
    '''Residual connection and layer norm'''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ResidualFeedForward0, self).__init__()
        self.size = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.sublayer = FeedForward0(d_model, d_ff, d_model, dropout=0.1)

    def forward(self, x):
        "Apply residual connection to any sublayer with the _same size_."
        return x + self.norm(self.dropout(self.sublayer(x)))

class ResidualSelfAttention0(nn.Module):
    '''
    Residual connection and layer norm with self attention
    Permutation EQUIvariant
    '''

    def __init__(self, heads, d_model, dropout=0.1):
        super(ResidualSelfAttention0, self).__init__()
        self.size = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.sublayer = lambda x: self.attn(x, x, x)
       
    def forward(self, x):
        assert(x.dim() == 3) # batchsize, setsize, features
        "Apply residual connection to any sublayer with the _same size_."
        return x + self.norm(self.dropout(self.sublayer(x)))

class DeepSet0(nn.Module):
    '''
    Simple Permutation Invariant Module
    https://arxiv.org/pdf/1703.06114.pdf 
    '''
    def __init__(self, phi, rho):
        super(DeepSet0, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        assert(x.dim() == 3) # batch, setsize, d_in
        return self.rho(self.phi(x).sum(dim=1))


'''
Main representation branches
'''

class FieldRepresentation0(nn.Module):
    '''
    `field/` 
    '''
    def __init__(self, d_out, s, dropout=0.1):
        super(FieldRepresentation0, self).__init__()

        
        d_field = s['fieldeffect']['embed_dim']
        d_weather = s['weather']['embed_dim']
        d_spikes = s['spikes']['embed_dim']
        d_toxicspikes = s['toxicspikes']['embed_dim']
        d_move = s['move']['embed_dim']

        self.d_out = d_out
        self.cat_dim = 25 * d_field + d_weather + 2 * d_spikes + 2 * d_toxicspikes
        self.cat_dim += 14 # _time fields

        # to reduce rel importance of twoturnmove
        self.twoturnmovedimred = FeedForward0(d_move, d_move, d_field, dropout=dropout)

        # final (after all field effects got concatenated)
        self.final = nn.Sequential(
            FeedForward0(self.cat_dim, self.cat_dim, d_out, dropout=dropout), 
            ResidualFeedForward0(d_out, d_out, dropout=dropout),
            ResidualFeedForward0(d_out, d_out, dropout=dropout),
        )

        
    def forward(self, x):

        x = x['field']
        h = [
             x['confusion'],
             x['confusionopp'],
             x['encore'],
             x['encore_time'],
             x['encoreopp'],
             x['encoreopp_time'],
             x['lightscreen'],
             x['lightscreen_time'],
             x['lightscreenopp'],
             x['lightscreenopp_time'],
             x['reflect'],
             x['reflect_time'],
             x['reflectopp'],
             x['reflectopp_time'],
             x['seed'],
             x['seedopp'],
             x['spikes'],
             x['spikesopp'],
             x['stealthrock'],
             x['stealthrockopp'],
             x['sub'],
             x['subopp'],
             x['tailwind'],
             x['tailwind_time'],
             x['tailwindopp'],
             x['tailwindopp_time'],
             x['taunt'],
             x['taunt_time'],
             x['tauntopp'],
             x['tauntopp_time'],
             x['torment'],
             x['torment_time'],
             x['tormentopp'],
             x['tormentopp_time'],
             x['toxicspikes'],
             x['toxicspikesopp'],
             x['trickroom'],
             x['trickroom_time'],
             x['twoturnmove'],
             self.twoturnmovedimred(x['twoturnmoveid']),
             x['twoturnmoveopp'],
             self.twoturnmovedimred(x['twoturnmoveoppid']),
             x['weather'],
             x['weather_time']
        ]

        h = torch.cat(h, dim=1)
        assert(h.shape[1] == self.cat_dim)

        # final
        h = self.final(h)
        return h



class MoveRepresentation0(nn.Module):
    '''
    Creates representation of `move_state` 
    '''
    def __init__(self, d_out, s, dropout=0.1):
        super(MoveRepresentation0, self).__init__()
        
        d_move = s['move']['embed_dim']
        d_move_type = s['move_type']['embed_dim']
        d_disabled = s['disabled']['embed_dim']

        self.move =      FeedForward0(d_move, d_move, d_out, dropout=dropout)
        self.move_type = FeedForward0(d_move_type, d_move_type, d_out, dropout=dropout)
        self.disabled =  FeedForward0(d_disabled, d_disabled, d_out, dropout=dropout)

        self.d_out = d_out
        self.cat_dim = 3 * d_out + 2

        self.final = nn.Sequential(
            FeedForward0(self.cat_dim, self.cat_dim, d_out),
            ResidualFeedForward0(d_out, d_out, dropout=dropout),
            ResidualFeedForward0(d_out, d_out, dropout=dropout),
        )

    def forward(self, x):
       
        h = [
             self.move(x['moveid']),
             self.move_type(x['movetype']),
             self.disabled(x['disabled']),
             x['maxpp'],
             x['pp']
        ]

        h = torch.cat(h, dim=1)
        assert(h.shape[1] == self.cat_dim)

        # final
        h = self.final(h)
        return h


class PokemonRepresentation0(nn.Module):
    '''
    Creates representation of `pokemon_state` 
    '''
    def __init__(self, d_out, s, dropout=0.1):
        super(PokemonRepresentation0, self).__init__()

        d_pokemon = s['pokemon']['embed_dim']
        d_type = s['type']['embed_dim']
        d_ability = s['ability']['embed_dim']
        d_item = s['item']['embed_dim']
        d_condition = s['condition']['embed_dim']
        d_alive = s['alive']['embed_dim']

        self.cat_dim = d_out # deep set move representation
        self.cat_dim += d_pokemon + 2 * d_type + d_ability + d_item + d_condition + d_alive # pokemon info
        self.cat_dim += 8 # ints in pokemon state
        self.cat_dim += 1 # imputed remaining health
        self.cat_dim += 7 # boosts

        # shared move representation learned for each move (permutation EQUIvariance)  
        self.move_embed = MoveRepresentation0(d_out, s)
        self.move_relate = ResidualSelfAttention0(heads=4, d_model=d_out, dropout=dropout)

        # relationship deep set function (permutation INvariance)      
        self.move_DS = DeepSet0(
            FeedForward0(d_out, d_out, d_out), # phi
            FeedForward0(d_out, d_out, d_out)) # rho
        
        self.final = nn.Sequential(
            FeedForward0(self.cat_dim, self.cat_dim, d_out),
            ResidualFeedForward0(d_out, d_out, dropout=dropout),
            ResidualFeedForward0(d_out, d_out, dropout=dropout),
        )

    def forward(self, pokemon, boosts):
        
        assert(boosts.dim() == 2 and boosts.shape[1] == 7)

        x = pokemon
        
        # order equivariant move representations
        moves = torch.stack([self.move_embed(x['moves'][i]) for i in range(4)], dim=1) # bs, moves, d
        
        moves_equivariant = self.move_relate(moves)
        
        # order invariant deep sets representation of all moves
        moves_invariant = self.move_DS(moves_equivariant)

        # pokemon representation
        h = [
             moves_invariant,
             x['alive'],
             x['baseAbility'],
             x['condition'],
             x['item'],
             x['pokemon_id'],
             x['pokemontype1'],
             x['pokemontype2'],
             x['stats']['atk'],
             x['stats']['def'],
             x['stats']['spa'],
             x['stats']['spd'],
             x['stats']['spe'],
             x['level'],
             x['hp'],
             x['stats']['max_hp'] - x['hp'], # remaining
             x['stats']['max_hp'],
             boosts.to(DEVICE)
        ]

        h = torch.cat(h, dim=1)
        assert(h.shape[1] == self.cat_dim)

        # final
        h = self.final(h)

        # return (hidden representation, equivariant move representations (for policy))
        return h, moves_equivariant

class PlayerRepresentation0(nn.Module):
    '''
    `player`/  and `opponent/` 
    '''
    def __init__(self, d_out, s, dropout=0.1):
        super(PlayerRepresentation0, self).__init__()

        # pokemon representations (shared/permutation equivariant for team pokemon)
        self.active_pokemon = PokemonRepresentation0(d_out, s, dropout=dropout)
        self.team_pokemon = PokemonRepresentation0(d_out, s, dropout=dropout)
        self.team_pokemon_relate = ResidualSelfAttention0(heads=4, d_model=d_out, dropout=dropout)

        # relationship deep set function (permutation INvariance)      
        self.move_DS = DeepSet0(
            FeedForward0(d_out, d_out, d_out), # phi
            FeedForward0(d_out, d_out, d_out)) # rho

        # team pokemon relationship deep set function (permutation INvariance)      
        self.team_DS = DeepSet0(
            FeedForward0(d_out, d_out, d_out), # phi
            FeedForward0(d_out, d_out, d_out)) # rho

        # final 
        self.final = nn.Sequential(
            FeedForward0(2 * d_out, 2 * d_out, d_out),
            ResidualFeedForward0(d_out, d_out, dropout=dropout),
            # ResidualFeedForward0(d_out, d_out, dropout=dropout),
        )
        
        
    def forward(self, x):
        # assume x is either player or opponent
        is_opp = 'oppaccuracy' in x['boosts']

        # determine correct boosts
        active_boosts = torch.cat([
            x['boosts']['oppaccuracy'] if is_opp else x['boosts']['accuracy'],
            x['boosts']['oppatk'] if is_opp else x['boosts']['atk'],
            x['boosts']['oppdef'] if is_opp else x['boosts']['def'],
            x['boosts']['oppevasion'] if is_opp else x['boosts']['evasion'],
            x['boosts']['oppspa'] if is_opp else x['boosts']['spa'],
            x['boosts']['oppspd'] if is_opp else x['boosts']['spd'],
            x['boosts']['oppspe'] if is_opp else x['boosts']['spe'],
        ], dim=1).float()

        team_boosts = torch.zeros(active_boosts.shape).float()

        # active pokemon representation
        # (invariant, equivariant)

        active_pokemon, moves_equivariant = self.active_pokemon(x['active'], active_boosts)

        # team pokemon representation (equivariant move reps are discarded for pokemon reps)
        # equivariant  
        team_pokemon = torch.stack([self.team_pokemon(x['team'][i], team_boosts)[0] for i in range(6)], dim=1)
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

class DeePikachu0(nn.Module):
    '''Value and Policy Net'''
    def __init__(self, state_embedding_settings, d_player=128, d_opp=64, d_field=32, dropout=0.1, softmax=True):
        super(DeePikachu0, self).__init__()

        self.d_player = d_player
        self.d_opp = d_opp
        self.d_field = d_field
        self.softmax = softmax

        d_hidden = d_player + d_opp + d_field
        
        
        self.state_embedding = State(state_embedding_settings)

        # major hidden state 
        self.player = PlayerRepresentation0(d_out=d_player, s=state_embedding_settings)

        self.opponent = PlayerRepresentation0(d_out=d_opp, s=state_embedding_settings)

        self.field = FieldRepresentation0(d_out=d_field, s=state_embedding_settings)

        self.combine = nn.Sequential(
            ResidualFeedForward0(d_hidden, d_hidden, dropout=dropout),
            ResidualFeedForward0(d_hidden, d_hidden, dropout=dropout),
        )

        # value function
        self.value_function = FeedForward0(d_hidden, d_hidden, 1, dropout=dropout)

        # Q function (heads 1 and 2) 
        self.q_combine_moves_hidden = nn.ModuleList([
            FeedForward0(d_player + d_hidden, d_player + d_hidden, d_player, dropout=dropout),
            FeedForward0(d_player + d_hidden, d_player + d_hidden, d_player, dropout=dropout)])

        self.q_combine_pokemon_hidden = nn.ModuleList([
            FeedForward0(d_player + d_hidden, d_player + d_hidden, d_player, dropout=dropout),
            FeedForward0(d_player + d_hidden, d_player + d_hidden, d_player, dropout=dropout)])
            
        self.q_function = nn.ModuleList([
            nn.Sequential(
                ResidualSelfAttention0(heads=4, d_model=d_player, dropout=dropout),
                FeedForward0(d_player, d_player, 1, dropout=dropout),
            ),
            nn.Sequential(
                ResidualSelfAttention0(heads=4, d_model=d_player, dropout=dropout),
                FeedForward0(d_player, d_player, 1, dropout=dropout),
            )])
        
    def forward(self, x):
        
        state = copy.deepcopy(x) # embedding is put inplace
        state = self.state_embedding(state)

        # player 
        player, moves_equivariant, team_pokemon_equivariant = self.player(state['player'])

        # opponent
        opponent, _, _ = self.opponent(state['opponent'])

        # field
        f = self.field(state)

        hidden = torch.cat([player, opponent, f], dim=1)

        # value function
        value = self.value_function(hidden).squeeze(dim=1).sigmoid() # - apparently sigmoid not done in practice

        # q function: self attend to different action options 
        moves_and_hidden = torch.cat(
            [moves_equivariant, 
             hidden.unsqueeze(1).repeat((1, 4, 1))], 
        dim=2)

        pokemon_and_hidden = torch.cat(
            [team_pokemon_equivariant, 
             hidden.unsqueeze(1).repeat((1, 6, 1))], 
        dim=2)

        all_actions_A =  torch.cat([
            self.q_combine_moves_hidden[0](moves_and_hidden), 
            self.q_combine_pokemon_hidden[0](pokemon_and_hidden)], dim=1)
        q_values_A = self.q_function[0](all_actions_A).squeeze(dim=2).sigmoid() #- apparently sigmoid not done in practice

        all_actions_B =  torch.cat([
            self.q_combine_moves_hidden[1](moves_and_hidden), 
            self.q_combine_pokemon_hidden[1](pokemon_and_hidden)], dim=1)
        q_values_B = self.q_function[1](all_actions_B).squeeze(dim=2)#.sigmoid() #- apparently sigmoid not done in practice

        return q_values_A, q_values_B, value
        


if __name__ == '__main__':

    '''
    Example
    '''

    example_state = state.create_2D_state(100)

    state_embedding_settings = {
        'pokemon' :     {'embed_dim' : 100, 'dict_size' : MAX_TOK_POKEMON},
        'type' :        {'embed_dim' : 50, 'dict_size' : MAX_TOK_TYPE},
        'move' :        {'embed_dim' : 50, 'dict_size' : MAX_TOK_MOVE},
        'move_type' :   {'embed_dim' : 50, 'dict_size' : MAX_TOK_MOVE_TYPE},
        'ability' :     {'embed_dim' : 10, 'dict_size' : MAX_TOK_ABILITY},
        'item' :        {'embed_dim' : 10, 'dict_size' : MAX_TOK_ITEM},
        'condition' :   {'embed_dim' : 10, 'dict_size' : MAX_TOK_CONDITION},
        'weather' :     {'embed_dim' : 10, 'dict_size' : MAX_TOK_WEATHER},
        'alive' :       {'embed_dim' : 10, 'dict_size' : MAX_TOK_ALIVE},
        'disabled' :    {'embed_dim' : 10, 'dict_size' : MAX_TOK_DISABLED},
        'spikes' :      {'embed_dim' : 10, 'dict_size' : MAX_TOK_SPIKES},
        'toxicspikes' : {'embed_dim' : 10, 'dict_size' : MAX_TOK_TOXSPIKES},
        'fieldeffect' : {'embed_dim' : 10, 'dict_size' : MAX_TOK_FIELD},
    }

    d_player = 128
    d_opp = 64
    d_field = 32
        
    model = DeePikachu0(state_embedding_settings, d_player=d_player, d_opp=d_opp, d_field=d_field)
    policy, value = model(copy.deepcopy(example_state))

    print('Shapes: ', value.shape, policy.shape, )
    print('V(s) = {}'.format(value[0].item()))
    print('pi(moves   |s) = {}'.format(policy[0][:4].cpu().detach().numpy()))
    print('pi(switches|s) = {}'.format(policy[0][4:].cpu().detach().numpy()))




