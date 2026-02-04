import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
import pdb
from model.modules import *

class MultimodalDecoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(MultimodalDecoderLayer, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn+1)
        self.norm = LayerNorm(size)
                
    def forward(self, b, ft, x):
        s_x = self.sublayer[0](x, lambda x: self.attn[0](x, x, x, b.trg_mask))
        s_x = self.sublayer[1](s_x, lambda s_x: self.attn[1](s_x, ft['encoded_his'], ft['encoded_his'], b.his_mask))
        s_x = self.sublayer[2](s_x,
                               lambda s_x: self.attn[2](s_x, ft['encoded_query'], ft['encoded_query'], b.query_mask))
        s_x = self.sublayer[3](s_x, lambda s_x: self.attn[3](s_x, ft['cap_ft'], ft['cap_ft'], b.query_mask))
        out_x = self.sublayer[4](s_x, lambda s_x: self.attn[4](s_x, ft['temporal_ft'], ft['temporal_ft'], b.query_mask))
        out_x = self.sublayer[5](out_x, self.ff)
        ft['decoded_text'] = self.norm(out_x)

        return ft
