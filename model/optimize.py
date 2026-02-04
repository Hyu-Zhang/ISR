import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb
from torchtext import data, datasets

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) * \
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, ae_generator, criterion, opt=None, l=1.0):
        self.generator = generator
        self.ae_generator = ae_generator
        self.criterion = criterion
        self.opt = opt
        self.l = l

    def __call__(self, x, batch):
        y = batch.trg_y
        norm = batch.ntokens
        loss = 0
        out = self.generator(x, batch)
        out_loss = self.criterion(out.contiguous().view(-1, out.size(-1)),
                              y.contiguous().view(-1)) / norm.float()
        loss += out_loss

        # ae_norm = batch.qntokens
        # cap_ae_out = self.ae_generator(x, 'cap_ft')
        # cap_ae_loss = self.criterion(cap_ae_out.contiguous().view(
        #     -1, out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
        # loss += cap_ae_loss

        # temporal_ae_out = self.ae_generator(x, 'temporal_ft')
        # temporal_ae_loss = self.criterion(temporal_ae_out.contiguous().view(
        #     -1, out.size(-1)), batch.query.contiguous().view(-1)) / ae_norm.float()
        # loss += temporal_ae_loss

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        losses = {}
        losses['out'] = out_loss.item() * norm.float()
        # losses['temporal_ae'] = temporal_ae_loss * ae_norm.float()

        return losses