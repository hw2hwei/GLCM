import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_Attn(nn.Module):
    def __init__(self,  
                 embed_dim, 
                 max_seq_length):
        super(Self_Attn, self).__init__()
        self.max_seq_length = max_seq_length
        self.self_attn = nn.MultiheadAttention(embed_dim, 1, dropout=0.1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


    def forward(self, query, key, value, mask=None):
        query = query.permute(1,0,2)
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        captions = self.self_attn(query, key, value, attn_mask=mask)[0].permute(1,0,2)
        captions = self.dropout(captions)

        return captions
        
    def sample(self, query, key, value):
        query = query.permute(1,0,2)
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        seq_embed = self.self_attn(query, key, value)[0].permute(1,0,2)
        seq_embed = self.dropout(seq_embed)

        return seq_embed

    def calc_attn(self, query, key, value, mask=None):
        query = query.permute(1,0,2)
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        seq_embed, attn = self.self_attn(query, key, value, attn_mask=mask)
        seq_embed = seq_embed.permute(1,0,2)
        seq_embed = self.dropout(seq_embed)

        return seq_embed, attn
