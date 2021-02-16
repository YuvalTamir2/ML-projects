# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:45:31 2021

@author: tamiryuv
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CaptchaModel(nn.Module):
    def __init__(self, NUM_CHARS):
        super(CaptchaModel, self).__init__()
        #### CONV ENCODER #####
        self.net = nn.Sequential(
            nn.Conv2d(3,128,kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2)),
            nn.Conv2d(128,64, kernel_size = (3,3), padding = (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2)))   ### 1,64,12,50
        ### reducing dim before RNN
#       self.fc = nn.Linear(768, 64)
#        self.dp = nn.Dropout(0.2)
        ### RNN ################
#       self.lstm = nn.GRU(64, 32, bidirectional = True, dropout=0.2, num_layers= 2, batch_first = True)
        self.for_tar = nn.Embedding(20,768)
        # self.src_words_embed = nn.Embedding(768, 768)
        # self.src_position_embed = nn.Embedding(20, 768)
        self.transformer = nn.Transformer(768,3,1,1,2, dropout=0.1)
        self.outs = nn.Linear(768, NUM_CHARS)
        
        
    def forward(self,images, targets = None,test = False):
        b,c,h,w = images.shape
        if test:
            tars = targets
        else:
            tars = targets[:,:-1]
            
        tar_b,tar_size = tars.shape
        # print(targets)
        ## images shape = [b,c,h,w]
        outs = self.net(images)
        x = outs.permute(0,3,1,2)  # [b,w,c,h]

        x = x.reshape(b,x.shape[1],-1) # [b,w,c*h]
        tar = self.for_tar(tars.long())
        targetss,x = tar.permute(1,0,2), x.permute(1,0,2)
        nopeak_mask = np.triu(np.ones((tar_size, tar_size)),k=1).astype('uint8')
        nopeak_mask = torch.from_numpy(nopeak_mask) == 0
        trg_mask = self.transformer.generate_square_subsequent_mask(tar_size) 
        # print(nopeak_mask)
        # print(trg_mask.shape)
        # print(x.shape,targetss.shape, nopeak_mask.shape)
        out = self.transformer(x,targetss, tgt_mask = trg_mask)#,src_key_padding_mask = src_padding_mask, tgt_mask = trg_mask)
        # print(out, out.shape)
        outs = self.outs(out)
        
       # outs = outs.repeat(4,1,1)
       # print(outs.shape)
        # if test:
        #     for i in range(1,targets.shape[0]):
        #         outs[0][i-1] += outs[0][i]
        if test:
            return outs
        else:# targets is not None:
       #     print(targets.shape)
            log_softmax_val = F.log_softmax(outs, 2)
       #     print(log_softmax_val.shape[2], log_softmax_val.size(2))
 #           print(log_softmax_val)
            input_length = torch.full(size = (b,), fill_value=log_softmax_val.shape[0], dtype = torch.int32)
            
            output_length = torch.full(size = (b,), fill_value = tars.shape[1], dtype = torch.int32)
            # print(input_length.shape, output_length.shape, targets[:,1:].shape,log_softmax_val.shape)
            loss = nn.CTCLoss(blank=0,zero_infinity = True)(log_softmax_val, targets[:,1:], input_length, output_length)
            # print(tars,targets[:,1:])
     #       print(log_softmax_val.shape[0])

            return outs, loss
            

    
        
        
            
        

# m = CaptchaModel(19)
# o = m(torch.rand(32,3,50,200),torch.randint(1,20,(32,5)))
# print(o[1])
