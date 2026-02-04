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

class Encoder(nn.Module):
    def __init__(self, size, nb_layers):
        super(Encoder, self).__init__()
        self.norm = nn.ModuleList()
        self.nb_layers = nb_layers
        for n in range(nb_layers):
            self.norm.append(LayerNorm(size))

    def forward(self, *seqs):
        output = []
        i=0
        seq_i=0
        while True:
            if isinstance(seqs[seq_i],list):
                output_seq = []
                for seq in seqs[seq_i]:
                    output_seq.append(self.norm[i](seq))
                    i+=1
                output.append(output_seq)
                seq_i+=1
            else:
                if seqs[seq_i] is None:
                    output.append(None)
                    seq_i += 1
                else:
                    output.append(self.norm[i](seqs[seq_i]))
                    i += 1
                    seq_i += 1
            if seq_i == len(seqs):
                break
        return output

class CapEncoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(CapEncoderLayer, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn + 1)

    def forward(self, c, ft, b):
        c = self.sublayer[0](c, lambda c: self.attn[0](c, c, c, b.query_mask))
        c = self.sublayer[1](c, lambda c: self.attn[1](c, ft['encoded_cap'], ft['encoded_cap'], b.cap_mask))
        c = self.sublayer[2](c, self.ff)
        return c

class VisEncoderLayer(nn.Module):
    def __init__(self, size, attn, nb_attn, ff, dropout):
        super(VisEncoderLayer, self).__init__()
        self.size = size
        self.attn = clones(attn, nb_attn)
        self.ff = clones(ff, 2)
        self.sublayer = clones(SublayerConnection(size, dropout), nb_attn + 2)

        # self.q_lstm = nn.LSTM(size, size, 1, batch_first=True)
        # self.v_lstm = nn.LSTM(size, size, 1, batch_first=True)
        # self.pooling = nn.AdaptiveAvgPool2d(16)


    def multi_head_attention(self, ft, b, weight):

        query, video = ft['hy_q'], ft['hy_v']
        q1 = self.sublayer[0](query, lambda query: self.attn[0](query, query, query, b.query_mask))
        q2 = self.sublayer[1](query, lambda query: self.attn[1](query, video, video, None))

        v1 = self.sublayer[2](video, lambda video: self.attn[2](video, query, query, b.query_mask))
        v2 = self.sublayer[3](video, lambda video: self.attn[3](video, video, video, None))
        # '''Global'''
        # ft['hy_q'] = q1 * weight[:, 0, 0].unsqueeze(1).unsqueeze(1).expand(q1.shape[0], q1.shape[1], q1.shape[2]) + q2 * weight[:, 0, 1].unsqueeze(1).unsqueeze(1).expand(q2.shape[0], q2.shape[1], q2.shape[2])
        # ft['hy_v'] = v1 * weight[:, 1, 0].unsqueeze(1).unsqueeze(1).expand(v1.shape[0], v1.shape[1], v1.shape[2]) + v2 * weight[:, 1, 1].unsqueeze(1).unsqueeze(1).expand(v1.shape[0], v1.shape[1], v1.shape[2])
        '''Local'''
        q1_weight = weight[:, :query.shape[1], :query.shape[1]].sum(dim=-1).unsqueeze(-1).expand(-1, -1, self.size)
        q2_weight = weight[:, :query.shape[1], query.shape[1]:].sum(dim=-1).unsqueeze(-1).expand(-1, -1, self.size)
        v1_weight = weight[:, query.shape[1]:, :query.shape[1]].sum(dim=-1).unsqueeze(-1).expand(-1, -1, self.size)
        v2_weight = weight[:, query.shape[1]:, query.shape[1]:].sum(dim=-1).unsqueeze(-1).expand(-1, -1, self.size)

        ft['hy_q'] = q1 * q1_weight + q2 * q2_weight
        ft['hy_v'] = v1 * v1_weight + v2 * v2_weight

        ft['hy_q'] = self.sublayer[4](ft['hy_q'], self.ff[0])
        ft['hy_v'] = self.sublayer[5](ft['hy_v'], self.ff[1])
        return ft

    def forward(self, ft, b):

        # q (batch, len, d_model)
        # vis (batch, temporal, spatial, d_model)
        ft['hy_q'] = ft['encoded_query']
        vis = ft['video_ft']
        # perm_vis = vis.transpose(1, 3)
        # # (batch, d_model, temporal, spatial)
        # # perm_vis = F.interpolate(perm_vis, scale_factor=0.5, recompute_scale_factor=True)
        # perm_vis = self.pooling(perm_vis)
        # # (batch, d_model, temporal/2, spatial/2)
        # perm_vis = perm_vis.transpose(1, 3)
        ft['hy_v'] = vis.reshape(vis.shape[0], vis.shape[1]*vis.shape[2], vis.shape[3])
        # (batch, temporal/2*spatial/2, d_model)
        '''Global'''
        # for i in range(3):
        #     q_output, (q_h_n, q_c_n) = self.q_lstm(ft['hy_q'])
        #     q_h_n = q_h_n.transpose(0, 1)
        #     # (batch, 1, d_model)
        #     v_output, (v_h_n, v_c_n) = self.v_lstm(ft['hy_v'])
        #     v_h_n = v_h_n.transpose(0, 1)
        #     # (batch, 1, d_model)
        #     con_h_n = torch.cat((q_h_n, v_h_n), dim=1)
        #     # (batch, 2, d_model)
        #     scores = torch.matmul(con_h_n, con_h_n.transpose(1, 2))
        #     weight = F.softmax(scores, dim=-1)
        #     # (batch, 2, 2)
        #     ft = self.multi_head_attention(ft, b, weight)

        '''Local'''
        for i in range(3):
            con_h_n = torch.cat((ft['hy_q'], ft['hy_v']), dim=1)
            # (batch, q_len+v_len, d_model)
            scores = torch.matmul(con_h_n, con_h_n.transpose(1, 2))
            weight = F.softmax(scores, dim=-1)
            # (batch, q_len+v_len, q_len+v_len)
            ft = self.multi_head_attention(ft, b, weight)
        out = ft['hy_q']

        return out

class Vislinear(nn.Module):
    def __init__(self, W, size):
        super(Vislinear, self).__init__()
        self.W = W
        self.in_norm = LayerNorm(size)

    def forward(self, b, ft):
        fts = nn.ReLU(inplace=True)(self.W(b.fts))
        fts = self.in_norm(fts)
        # fts = fts.transpose(1, 2)
        ft['video_ft'] = fts
        return ft