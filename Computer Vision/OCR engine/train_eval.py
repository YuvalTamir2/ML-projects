# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:44:42 2021

@author: tamiryuv
"""
import torch
from model import CaptchaModel
import DataSets


#Captcha_model = CaptchaModel(DataSets.NUM_CHARS)


def train_loop(train_loader, model, optimizer):
    model.to(DataSets.device)
    model.train()
    epoch_loss = 0
    for idx, (image, lab) in enumerate(train_loader):
        image, lab = image.to(DataSets.device), lab.to(DataSets.device)
        losss = 0
        # for i in range(1,lab.shape[1]):
            # optimizer.zero_grad()
            # outs, loss = model(image,lab[:,:i+1])

            # losss += loss
        optimizer.zero_grad()
        outs,losss = model(image,lab)
        losss.backward()
        optimizer.step()
        epoch_loss += losss.item()
  #      print('batch {} loss {}'.format(idx, loss.item()))
        
    return epoch_loss / len(train_loader)


# train_loss = train_loop(DataSets.train_loader, Captcha_model, optimizer)

def eval_loop(train_loader, model, optimizer):
    model.to(DataSets.device)
    model.eval()
    epoch_loss = 0
    preds = []
    for idx, (image, lab) in enumerate(train_loader):
        image, lab = image.to(DataSets.device), lab.to(DataSets.device)
        with torch.no_grad():
            

                outs, loss = model(image, lab)
            # outs, loss = model(image, lab[:,:-1])
                preds.append(outs)
                epoch_loss += loss.item()
#        print('batch {} loss {}'.format(idx, loss.item()))
        
    return preds, epoch_loss / len(train_loader)

#train_loss = train_loop(DataSets.train_loader, Captcha_model, optimizer)
#pred,eval_loss = eval_loop(DataSets.train_loader, Captcha_model, optimizer)


def decode_preds(pred,lbl_enc):
    preds = pred.permute(1,0,2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k -= 1
            if k == -1:
                temp.append("§")
            else:
                p = lbl_enc.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(tp)
    return cap_preds