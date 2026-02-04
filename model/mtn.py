import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable
from data.data_utils import *
from model.modules import *
from model.decoder import *
from model.encoder import *
from model.generator import *
from model.modules import *

class EncoderDecoder(nn.Module):
    def __init__(self,  text_encoder, vis_linear, c_layer, v_layer, decode_layer, query_embed, tgt_embed, generator, auto_encoder_generator=None):
        super(EncoderDecoder, self).__init__()
        self.vis_linear = vis_linear
        self.text_encoder = text_encoder
        self.query_embed = query_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.auto_encoder_generator=auto_encoder_generator
        self.c_layer = c_layer
        self.v_layer = v_layer
        self.decode_layer = decode_layer

        self.cap_norm = LayerNorm(c_layer.size)
        self.vis_norm = LayerNorm(v_layer.size)
        # self.vc_combine_W = nn.Linear(v_layer.size*3, 2)

    def encode(self, b):
        ft = {}
        ft = self.vis_linear(b, ft)
        encoded_query, encoded_cap, encoded_his = self.text_encoder(self.query_embed(b.query), self.query_embed(b.cap), self.query_embed(b.his))
        ft['encoded_query'] = encoded_query
        ft['encoded_his'] = encoded_his
        ft['encoded_cap'] = encoded_cap

        c = self.c_layer(encoded_query, ft, b)
        ft['cap_ft'] = self.cap_norm(c)

        vis = self.v_layer(ft, b)
        ft['temporal_ft'] = self.vis_norm(vis)

        # temp = torch.cat([ft['encoded_query'], ft['cap_ft'], ft['temporal_ft']], dim=-1)
        # combine_score = F.softmax((self.vc_combine_W(temp)), dim=-1)
        # ft['encoded_ft'] = combine_score[:,:,0].unsqueeze(-1)*ft['temporal_ft'] + \
        #     combine_score[:,:,1].unsqueeze(-1)*ft['cap_ft']

        return ft

    def decode(self, b, ft):
        encoded_tgt = self.tgt_embed(b.trg)
        ft['encoded_tgt'] = encoded_tgt
        ft = self.decode_layer(b, ft, encoded_tgt)

        return ft

    def forward(self, b):
        ft = self.encode(b)
        ft = self.decode(b, ft)

        return ft

def make_model(src_vocab, tgt_vocab, d_model=128, d_ff=2048, h=8, dropout=0.2, ft_sizes=None):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    query_embed = [Embeddings(d_model, src_vocab), c(position)]
    query_embed = nn.Sequential(*query_embed)
    tgt_embed = query_embed

    pointer_attn = nn.ModuleList()
    ptr_ft_ls = ["query","cap"]
    for ft in ptr_ft_ls:
        pointer_attn.append(MultiHeadedAttention(1, d_model, dropout=0))
    generator = MultiPointerGenerator(d_model, tgt_embed[0].lut.weight, pointer_attn, ptr_ft_ls)

    text_encoder=Encoder(d_model, nb_layers=3)
    vid_W = nn.Linear(ft_sizes[0], d_model)
    vis_linear = Vislinear(c(vid_W), d_model)
    ae_generator = Generator(d_model, tgt_vocab, tgt_embed[0].lut.weight)
    c_layer = CapEncoderLayer(d_model, c(attn), 2, c(ff), dropout)
    v_layer = VisEncoderLayer(d_model, c(attn), 4, c(ff), dropout)
    decode_layer = MultimodalDecoderLayer(d_model, c(attn), 5, c(ff), dropout)

    model = EncoderDecoder(
          text_encoder=text_encoder,
          vis_linear=vis_linear,
          c_layer=c_layer,
          v_layer=v_layer,
          decode_layer=decode_layer,
          query_embed=query_embed,
          tgt_embed=tgt_embed,
          generator=generator,
          auto_encoder_generator=ae_generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
