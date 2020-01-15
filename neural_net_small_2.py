

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
def make_f_activation():
    return copy.deepcopy(nn.LeakyReLU())

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
       
    def forward(self, x):    
        return x

def make_identity():
    return Identity()

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
MAX_TOK_MOVE_CATEGORY = 7

HP_SCALE = 10

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

class SingleEmbedding2(torch.nn.Module):
    def __init__(self, setting):
        super(SingleEmbedding2, self).__init__()
        self.embedding = torch.nn.Embedding(
            setting['dict_size'], 
            setting['embed_dim'])
        self.dim = setting['embed_dim']

    def forward(self, pokemon_state):    
        x = self.embedding(pokemon_state)
        return x

class State2(torch.nn.Module):
    '''Returns original dict with all tokens replaced by its embeddings'''
    def __init__(self, settings):
      
        super(State2, self).__init__()

        # `key`s in dict associated with desired embedding
        self.type_fields = set(['pokemontype1', 'pokemontype2', 'movetype'])
        self.move_fields = set(['moveid'])
        self.condition_fields = set(['condition'])
        self.move_category_fields = set(['category'])
   
        # manual impuations
        self.opponent_field = set(['opponent'])

        # main embeddings
        self.type_embedding = SingleEmbedding2(settings['type'])
        self.move_embedding = SingleEmbedding2(settings['move'])
        self.condition_embedding = SingleEmbedding2(settings['condition'])
        self.move_category_embedding = SingleEmbedding2(settings['move_category'])

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
                elif key in self.move_category_fields:
                    x[key] = self.move_category_embedding(torch.tensor(value, dtype=torch.long, device=DEVICE))
               
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

        self.d_k = d_model // h
        self.h = h

        self.W_q = nn.Linear(d_model, self.h * self.d_k)
        self.W_k = nn.Linear(d_model, self.h * self.d_k)
        self.W_v = nn.Linear(d_model, self.h * self.d_k)
        self.W_o = nn.Linear(self.h * self.d_k, d_model, bias=False) # bias false s.t. masking of dead pokemon correct (I think)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def __attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        # feature dimension
        d_k = query.size(-1)   

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)        

        if mask is not None:
            # mask key sequence dim of dead pokemon with -inf
            # that way for every pok i, attention at j is 0 if pok j is dead
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, -1e9) 

            # mask value sequence dim of dead pokemon with 0
            # that way, when value matrix is dotted with attention matrix, resulting vectors for dead pokemon are 0 since no bias for W_o 
            
            # not doing that for now since option is masked anyway (not 100 sure itd be correct)
            # value = value.masked_fill(mask.unsqueeze(1).unsqueeze(3) == 0, 0.0) 

        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        
    def forward(self, query, key, value, mask=None):
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
        x, self.attn = self.__attention(query, key, value, dropout=self.dropout, mask=mask)
        
        # # 4) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(batch_size, -1, self.h * self.d_k)

        x = self.W_o(x)
        return x


class SelfAttention2(nn.Module):
    '''
    Self attention
    Permutation EQUIvariant
    '''

    def __init__(self, heads, d_model, dropout=0.0):
        super(SelfAttention2, self).__init__()
        self.size = d_model
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.sublayer = lambda x, mask: self.attn(x, x, x, mask=mask)
       
    def forward(self, x, mask=None): #, active):
        assert(x.dim() == 3) # batchsize, setsize, feature
        return self.sublayer(x, mask)

class DeepSet2(nn.Module):
    '''
    Simple Permutation Invariant Module
    https://arxiv.org/pdf/1703.06114.pdf 
    '''
    def __init__(self, phi, rho):
        super(DeepSet2, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x, mask_=None):
        assert(x.dim() == 3) # batch, setsize, d_in       
        rep = self.phi(x)

        # mask dead pokemon (mask is (batchsize, setsize))
        # divide by # pok alive to be invariant to # alive
        if mask_ is not None:
            # mask pokemon as `dead`
            mask = copy.deepcopy(mask_).to(DEVICE)
            mask[:, 0] = 0
            alive_ctr = mask.sum(dim=1)
            # ctr>= 1 o/w breaks at the end of game (when no alive poks) 
            alive_ctr = alive_ctr.masked_fill(alive_ctr == 0, 1.0).unsqueeze(1).unsqueeze(2).float()
            mask = mask.unsqueeze(2)
            rep = rep.masked_fill(mask == 0, 0.0) / alive_ctr
        
        return self.rho(rep.sum(dim=1))


'''
Main representation branches
'''

class MoveRepresentation2(nn.Module):
    '''
    Creates representation of `move_state` 
    '''
    def __init__(self, s, move_identity, layer_norm, dropout=0.0):
        super(MoveRepresentation2, self).__init__()
        
        self.d_move = s['move']['embed_dim']
        self.d_move_type = s['type']['embed_dim']
        self.d_move_category = s['move_category']['embed_dim']

        self.cat_dim = self.d_move + self.d_move_type + self.d_move_category + 1 # accuracy
        self.d_out = 2 * self.cat_dim if not move_identity else self.cat_dim # define this to be out dim

        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.d_out), make_f_activation(),
            nn.Linear(self.d_out, self.d_out), nn.LayerNorm(self.d_out) if layer_norm else make_identity()
        ) if not move_identity else make_identity()


    def forward(self, x):
        h = [
             x['moveid'],
             x['movetype'],
             x['category'],
             x['accuracy'] / 20 # else initialization is extreme
        ]

        h = torch.cat(h, dim=1)
        assert(h.shape[1] == self.cat_dim)

        # final
        h = self.final(h)
        return h, x['movetype']

class PokemonRepresentation2(nn.Module):
    '''
    Creates representation of `pokemon_state` 
    '''
    def __init__(self, s, d_pokemon, move_identity, layer_norm, dropout=0.0, attention=False):
        super(PokemonRepresentation2, self).__init__()

        self.d_type = s['type']['embed_dim']
        self.d_condition = s['condition']['embed_dim']

        # shared move representation learned for each move (permutation EQUIvariance)  
        self.move_representation = MoveRepresentation2(s, move_identity=move_identity, layer_norm=layer_norm, dropout=dropout)
        self.d_move = 2 * self.move_representation.d_out  # deep set move representation

        self.move_relate = SelfAttention2(heads=4, d_model=self.move_representation.d_out, dropout=dropout) if attention else make_identity()

        self.cat_dim = self.d_move # collective move representation
        self.cat_dim += 2 * self.d_type + self.d_condition  # pokemon info
        self.cat_dim += 6 # ints in pokemon state

        self.d_out = d_pokemon # define this to be out dimension

        # relationship deep set function (permutation INvariance)      
        self.move_DS = DeepSet2(
            nn.Sequential( # phi
                nn.Linear(self.d_move, self.d_move), make_f_activation(), 
                nn.Linear(self.d_move, self.d_move)), 
            nn.Sequential(  # rho
                nn.Linear(self.d_move, self.d_move), make_f_activation(), 
                nn.Linear(self.d_move, self.d_move))) 
        
        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.cat_dim), make_f_activation(),
            nn.Linear(self.cat_dim, self.d_out), nn.LayerNorm(self.d_out) if layer_norm else make_identity(), 
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
        moves_types_equivariant = torch.stack(moves_types, dim=1)  # bs, moves, d
        
        # cat representation with attention representation
        moves_equivariant = torch.cat(
            [moves, self.move_relate(moves)], dim=2)
        
        # order invariant deep sets representation of all moves
        moves_invariant = self.move_DS(moves_equivariant)

        # pokemon representation
        max_hp = copy.deepcopy(x['stats']['max_hp'])
        max_hp = max_hp.masked_fill(max_hp == 0, 1.0).float() # so no div by 0 if maxhp = 0
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

            

class PlayerRepresentation2(nn.Module):
    '''
    `player`/  and `opponent/` 
    '''
    def __init__(self, d_out, s, d_pokemon, move_identity, layer_norm, dropout=0.0, attention=False):
        super(PlayerRepresentation2, self).__init__()

        # pokemon representations (shared/permutation equivariant for team pokemon)
        self.d_pokemon = 2 * d_pokemon
        self.active_pokemon = PokemonRepresentation2(s, d_pokemon=d_pokemon, move_identity=move_identity, dropout=dropout, attention=attention, layer_norm=layer_norm)
        self.team_pokemon = PokemonRepresentation2(s, d_pokemon=d_pokemon, move_identity=move_identity, dropout=dropout, attention=attention, layer_norm=layer_norm)

        # dims
        assert(self.active_pokemon.d_out == self.team_pokemon.d_out 
            and self.active_pokemon.d_move == self.team_pokemon.d_move)
        self.d_move = self.active_pokemon.d_move 
        self.d_out = d_out
        self.cat_dim = d_pokemon + 2 * d_pokemon # active + team

        self.team_pokemon_relate = SelfAttention2(heads=4, d_model=d_pokemon, dropout=dropout) if attention else make_identity()

        # team pokemon relationship deep set function (permutation INvariance)    
        self.team_DS = DeepSet2(
            nn.Sequential( # phi
                nn.Linear(self.d_pokemon, self.d_pokemon), make_f_activation(), 
                nn.Linear(self.d_pokemon, self.d_pokemon)), 
            nn.Sequential(  # rho
                nn.Linear(self.d_pokemon, self.d_pokemon), make_f_activation(),
                nn.Linear(self.d_pokemon, self.d_pokemon))) 
            
        # final         
        self.final = nn.Sequential(
            nn.Linear(self.cat_dim, self.cat_dim), make_f_activation(),
            nn.Linear(self.cat_dim, self.d_out), nn.LayerNorm(self.d_out) if layer_norm else make_identity()
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
            poke_rep, _, _, _, poke_types, poke_hp = self.team_pokemon(x['team'][i])
            team_pokemon.append(poke_rep)
            team_pokemon_types1.append(poke_types[0])
            team_pokemon_types2.append(poke_types[1])
            team_pokemon_hp.append(poke_hp)

        team_pokemon = torch.stack(team_pokemon, dim=1)  # bs, pokemon, d
        team_pokemon_types1 = torch.stack(team_pokemon_types1, dim=1)  # bs, pokemon, d
        team_pokemon_types2 = torch.stack(team_pokemon_types2, dim=1)  # bs, pokemon, d
        team_pokemon_hp = torch.stack(team_pokemon_hp, dim=1)  # bs, pokemon, d
        
        # get alive status for masking in attention and deep set
        self.team_alive = torch.cat([x['team'][i]['alive'] for i in range(6)], dim=1)
        self.team_alive = self.team_alive.long().to(DEVICE)

        # cat representation with attention representation
        team_pokemon_equivariant = torch.cat(
            [team_pokemon,  self.team_pokemon_relate(team_pokemon, self.team_alive)], dim=2)  

        # invariant
        team = self.team_DS(team_pokemon_equivariant, self.team_alive)

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



'''
Final net
'''

class SmallDeePikachu2(nn.Module):
    '''Value and Policy Net'''

    def __init__(self, state_embedding_settings, hidden_layer_settings, layer_norm=False, move_identity=False, dropout=0.0, attention=False):
        super(SmallDeePikachu2, self).__init__()

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

        self.d_sim = self.d_type // 2 # arbitrary
        self.d_sim_opp_out = self.sim_heads_opp * self.sim_heads_opp
        self.d_sim_p_out = self.sim_heads_p * self.sim_heads_p

        # (player) + (opponent) + (similarity of four active type pairs) + (stats of both players active) + (hp summaries)
        self.d_hidden_in = self.d_player + self.d_opp + \
            4 * self.d_sim_opp_out + 12 + 4

        self.state_embedding = State2(state_embedding_settings)
        self.state_embedding_settings = state_embedding_settings

        # major hidden state 
        self.player = PlayerRepresentation2(
            d_out=self.d_player, s=state_embedding_settings, d_pokemon=hidden_layer_settings['pokemon_hidden'], 
            move_identity=move_identity, attention=attention, dropout=dropout, layer_norm=layer_norm)

        self.opponent = PlayerRepresentation2(
            d_out=self.d_opp, s=state_embedding_settings, d_pokemon=hidden_layer_settings['pokemon_hidden'], 
            move_identity=move_identity, attention=attention, dropout=dropout, layer_norm=layer_norm)

        self.d_move = self.player.d_move
        self.d_pokemon = self.player.d_pokemon

        self.hidden_reduce = nn.Sequential(
            nn.Linear(self.d_hidden_in, self.d_hidden_in), make_f_activation(),
            nn.Linear(self.d_hidden_in, self.d_context), nn.LayerNorm(self.d_context) if layer_norm else make_identity(),
        )
       

        # similarity metrics
        # A = player moves - opp active
        # B = player team  - opp active
        # C = player moves - player active
        self.A_p = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim) # p move - opp active
        self.A_opp = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim) 
        
        self.B_p = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim) # p team - opp active
        self.B_opp = nn.Linear(self.d_type, self.sim_heads_opp * self.d_sim)

        self.C_p_move = nn.Linear(self.d_type, self.sim_heads_p * self.d_sim) # p move - p active
        self.C_p_active = nn.Linear(self.d_type, self.sim_heads_p * self.d_sim)

        # value function
        self.value_function = nn.Sequential(
                nn.Linear(self.d_context, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1), 
            )
        
        # Q function (heads 1 and 2) 
        self.d_q_move_in = self.d_move + self.d_context + 2 * self.d_sim_opp_out + 2 * self.d_sim_p_out + 4 # hp summaries
        self.d_q_team_in = self.d_pokemon + self.d_context + 4 * self.d_sim_opp_out + 5 # team i hp + hp summaries
        self.q_combine_moves_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_q_move_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1),
            ),
            nn.Sequential(
                nn.Linear(self.d_q_move_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1),
            )])
        
        self.q_combine_pokemon_context = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_q_team_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, 1),
            ),
            nn.Sequential(
                nn.Linear(self.d_q_team_in, self.d_context), make_f_activation(),
                nn.Linear(self.d_context, self.d_context), make_f_activation(),
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

    def __dot_similarity(self, k, v, heads):
        # compute batched dot product and rescale heads
        batch_size = k.shape[0]
        out = torch.matmul(k, v.transpose(-2, -1)) \
            / math.sqrt(self.d_sim)
        return out.view( batch_size, -1, 
            heads * heads).contiguous()

        
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

        # similarity metrics comparing (looks more than it is)
        # player active move types - opponent active pokemon types
        # player team pokemon types - opponent active pokemon types
        mv_types = player_moves_types_equivariant
        team_type1 = player_team_pokemon_types1
        team_type2 = player_team_pokemon_types2
        p_active_type1 = player_active_type1.unsqueeze(1)
        p_active_type2 = player_active_type2.unsqueeze(1)

        opp_active_type1 = opponent_active_type1.unsqueeze(1)
        opp_active_type2 = opponent_active_type2.unsqueeze(1)

        batch_size = mv_types.shape[0]

        # perform all the linear operations (in batch from d_type -> h * d_similarity) and split into h heads
        move_p = self.A_p(mv_types).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        move_opp1 = self.A_opp(opp_active_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        move_opp2 = self.A_opp(opp_active_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)

        team_p1 = self.B_p(team_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        team_p2 = self.B_p(team_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        team_opp1 = self.B_opp(opp_active_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        team_opp2 = self.B_opp(opp_active_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)
        active_p1 = self.B_p(p_active_type1).view(batch_size, -1, self.sim_heads_opp, self.d_sim) # comp active as well 
        active_p2 = self.B_p(p_active_type2).view(batch_size, -1, self.sim_heads_opp, self.d_sim)

        stab_move = self.C_p_move(mv_types).view(batch_size, -1, self.sim_heads_p, self.d_sim)
        stab_active1 = self.C_p_active(p_active_type1).view(batch_size, -1, self.sim_heads_p, self.d_sim)
        stab_active2 = self.C_p_active(p_active_type2).view(batch_size, -1, self.sim_heads_p, self.d_sim)
              
        # # compute similarity scores, and concat using .view()
        # A
        mv_scores1 = self.__dot_similarity(move_p, move_opp1, self.sim_heads_opp)
        mv_scores2 = self.__dot_similarity(move_p, move_opp2, self.sim_heads_opp)

        # B
        team_scores11 = self.__dot_similarity(team_p1, team_opp1, self.sim_heads_opp)
        team_scores12 = self.__dot_similarity(team_p1, team_opp2, self.sim_heads_opp)
        team_scores21 = self.__dot_similarity(team_p2, team_opp1, self.sim_heads_opp)
        team_scores22 = self.__dot_similarity(team_p2, team_opp2, self.sim_heads_opp)
        
        # type scores: p active - opp active
        active_scores = torch.cat([
            self.__dot_similarity(active_p1, team_opp1, self.sim_heads_opp).squeeze(1),
            self.__dot_similarity(active_p1, team_opp2, self.sim_heads_opp).squeeze(1),
            self.__dot_similarity(active_p2, team_opp1, self.sim_heads_opp).squeeze(1),
            self.__dot_similarity(active_p2, team_opp2, self.sim_heads_opp).squeeze(1),
        ], dim=1) 
        
        # C
        mv_scores_own1 = self.__dot_similarity(stab_move, stab_active1, self.sim_heads_p)
        mv_scores_own2 = self.__dot_similarity(stab_move, stab_active2, self.sim_heads_p)


        # combine hidden representations of 
        # player, opponent, active sims, hps into context 
        hidden = torch.cat([
            player, 
            player_active_stats, 
            active_scores, 
            opponent,
            opponent_active_stats,
            player_active_hp,
            opponent_active_hp,
            player_team_hp_ave,
            opponent_team_hp_ave,
        ], dim=1)

        context = self.hidden_reduce(hidden)

        # value function
        value = self.value_function(context).squeeze(dim=1)

        # q function (add similarity scores of types)
        moves_and_context = torch.cat(
            [player_moves_equivariant,
             mv_scores1, mv_scores2,  # 2 * 4 metrics of similarity (comp w 2x opp type)
             mv_scores_own1, mv_scores_own2, # 2 * 4 metrics of similarity(comp w 2x own type),
             player_active_hp.unsqueeze(1).repeat((1, 4, 1)), # own hp
             player_team_hp_ave.unsqueeze(1).repeat((1, 4, 1)),
             opponent_active_hp.unsqueeze(1).repeat((1, 4, 1)), # opp hp
             opponent_team_hp_ave.unsqueeze(1).repeat((1, 4, 1)),
             context.unsqueeze(1).repeat((1, 4, 1))
             ], 
        dim=2)

        pokemon_and_context = torch.cat(
            [player_team_equivariant,
             team_scores11, team_scores12, team_scores21, team_scores22, # 4 * 4 metrics of similarity (comp 2x own type w 2x opp type)
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

    # for i in range(34):
    #     # example_state = state.create_2D_state(2)
    #     example_state = torch.load(f'output/state_{i}.pt')

    #     out = p1net(copy.deepcopy(example_state))

    #     qa, qb, v = out

    #     print('Q A: ', qa[0])
    #     print('Q B: ', qb[1])
    #     print('V  : ', v[2])



    # increase hp scale
    # skip hps to the end 
    #   context: active (already there) sum of team
    #   after context: active, sum of team
    #   to each team pokemon: hp of that pokemon
    '''

    Skip the following variables forward to final prediction layers and context:
    Own hp percentage * 100
    Opponent hp percentage * 100
    Average own team total percentage* 100
    Average opponent team percentage hp * 100
    Hp of pokemon to each pokemon representation unprocessed
    '''


