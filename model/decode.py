import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb
import re
from collections import Counter
from nltk.util import ngrams
from data.data_utils import *

def beam_search_decode(model, batch, max_len, start_symbol, unk_symbol, end_symbol, pad_symbol, beam=5, penalty=1.0,
                       nbest=5, min_len=1, dec_eos=False):

    ft = model.encode(batch)
    ds = torch.ones(1, 1).fill_(start_symbol).long()
    hyplist = [([], 0., ds)]
    best_state = None
    comp_hyplist = []
    for l in range(max_len):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            batch.trg = Variable(st).cuda()
            batch.trg_mask = Variable(subsequent_mask(st.size(1)).long()).cuda()
            batch.trg_mean_mask = torch.ones(batch.trg.shape).long().cuda()

            output = model.decode(batch, ft)
            output['decoded_text'] = output['decoded_text'][:,-1].unsqueeze(1)
            output['encoded_tgt'] = output['encoded_tgt'][:,-1].unsqueeze(1)

            logp = model.generator(output, batch)

            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp: 
                    best_state = new_lp
            count = 1 
            for o in np.argsort(lp_vec)[::-1]:
                if dec_eos and (o == unk_symbol):
                    continue 
                if not dec_eos and (o == unk_symbol or o == end_symbol):
                    continue 
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1,1).long().fill_(int(o))], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                    else:
                        break
                else: 
                    new_st = torch.cat([st, torch.ones(1,1).long().fill_(int(o))], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h:h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
            
    if len(comp_hyplist) > 0: 
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0)], None