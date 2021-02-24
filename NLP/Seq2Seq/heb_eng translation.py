
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:39:29 2020

@author: tamiryuv
"""
### Seq2Seq with transformers!!!

import torch.nn as nn
import torch
import torchtext
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchtext.data import Field, BucketIterator, TabularDataset
import random


path = r'Eng_Heb.csv'

tokenizer = lambda x: x.split()

SRC = Field(tokenize = tokenizer,
            init_token='<sos>',
            eos_token='<eos>',
            include_lengths=True)

TRG = Field(tokenize = tokenizer,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

fields = [('trg',TRG), ('src',SRC)]

train = TabularDataset(path = path, format = 'csv', fields = fields)

SRC.build_vocab(train,max_size=10000,min_freq = 2)
TRG.build_vocab(train,max_size=10000,min_freq= 2)

train_iter = BucketIterator(train, batch_size = 64,
                            sort_within_batch=True,
                            sort_key = lambda x: len(x.src))


class Transformer(nn.Module):
    def __init__(self,src_vocab_len,
                 trg_vocab_len,
                 embed_size,
                 num_encoder_layers,
                 num_decoder_layers,
                 max_len,
                 num_heads,
                 forward_expansion,
                 src_pad_idx):
        super(Transformer,self).__init__()
        self.src_pad_idx = src_pad_idx
        
        self.src_words_embed = nn.Embedding(src_vocab_len, embed_size)
        self.src_position_embed = nn.Embedding(max_len, embed_size)
        self.trg_words_embed = nn.Embedding(trg_vocab_len, embed_size)
        self.trg_posiotion_embed = nn.Embedding(max_len, embed_size)
        
        self.transformer = nn.Transformer(embed_size,num_heads,num_encoder_layers,num_decoder_layers,forward_expansion)
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_len)
        
    def create_src_mask(self,src):
        src_mask = src.transpose(0,1) == self.src_pad_idx
        return src_mask
    
    def forward(self,src,trg):
        
        src_seq_len,src_batch = src.shape
        trg_seq_len,trg_batch = trg.shape #[h,b]
        
        src_position = torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len,src_batch)
        trg_position = torch.arange(0,trg_seq_len).unsqueeze(1).expand(trg_seq_len,trg_batch)
    
        embed_src = self.src_words_embed(src) + self.src_position_embed(src_position)
        embed_trg = self.trg_words_embed(trg) + self.trg_posiotion_embed(trg_position)

        src_padding_mask = self.create_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len)     
        print(embed_src.shape,embed_trg.shape)
        out = self.transformer(embed_src,embed_trg,src_key_padding_mask = src_padding_mask, tgt_mask = trg_mask)
        print(trg_mask)
        
        out = self.fc_out(out)
        
        return out
    
#train HP:

EPOCHES = 3
lr = 1e-3
batch_size = 64

# model HP:
src_vocab_len = len(SRC.vocab)
trg_vocab_len = len(TRG.vocab)
embed_size = 256
num_heads = 8
num_decoder_layers = 3
num_encoder_layers = 3
max_len = 100
src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
forward_expansion = 4        
        
LongLiveTransformer = Transformer(src_vocab_len, trg_vocab_len, embed_size, num_encoder_layers, num_decoder_layers, max_len, num_heads, forward_expansion, src_pad_idx)     
initial = LongLiveTransformer.state_dict()        
optimizer = torch.optim.Adam(params = LongLiveTransformer.parameters(),lr = lr)
trg_pad_idx = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index= trg_pad_idx)

def train(LongLiveTransformer,train_set):
    LongLiveTransformer.train()
    epoch_loss = 0
    best_state_dict = None
    for batch in train_set:
        src,src_len = batch.src
        trg = batch.trg
        outputs = LongLiveTransformer(src,trg[:-1])
        out_dim_size = outputs.shape[-1]
        outputs = outputs.reshape(-1,out_dim_size)
        trg = trg[1:].reshape(-1)
        
        loss = criterion(outputs, trg)
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        best_wts = LongLiveTransformer.state_dict()
    LongLiveTransformer.load_state_dict(best_wts)    
    return best_wts,epoch_loss / len(train_iter)


wts,loss = train(LongLiveTransformer,train_set = train_iter)

def translate(sentence, model, max_len = 50):
    model.eval()
    assert type(sentence)!='str', 'This is not a valid sentence..'
    
    sen_tokens = tokenizer(sentence)
    # adding the start and end tokens :
    sen_tokens.insert(0, '<sos>')
    sen_tokens.append('<eos>')
    src_tokens = [SRC.vocab.stoi[i] for i in sen_tokens]
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(1)
    
    trg_tensor = torch.LongTensor(TRG.vocab.stoi['<sos>']).unsqueeze(1)
    outs = [TRG.vocab.stoi['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(outs).unsqueeze(1)
        with torch.no_grad():
            preds = model(src_tensor,trg_tensor)
        
        best_guss = preds.argmax(2)[-1,:].item()
        outs.append(best_guss)
        
        if best_guss == TRG.vocab.stoi['<eos>']:
            break
        
        translated = [TRG.vocab.itos[i] for i in outs]
        
    return ' '.join(translated)
