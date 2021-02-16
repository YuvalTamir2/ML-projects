# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:11:30 2021

@author: tamiryuv
"""
import train_eval
import DataSets
import torch
import model
from torch.utils.data import DataLoader
import cv2
import torchvision


# train_loader = DataLoader(DataSets.train_dataset, batch_size=32, shuffle=True)
# test_loader =DataLoader(DataSets.test_dataset, batch_size=32, shuffle = False)

EPOCHES = 50
Captcha_model = model.CaptchaModel(DataSets.NUM_CHARS)
l = Captcha_model.state_dict()
init = l['outs.weight'][0][-1].item()
print(init)
optimizer = torch.optim.Adam(params = Captcha_model.parameters(), lr = 3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True)

min_loss = 10
for epoch in range(1,EPOCHES):
    train_loss = train_eval.train_loop(DataSets.train_loader, Captcha_model, optimizer)
    preds, eval_loss = train_eval.eval_loop(DataSets.test_loader, Captcha_model, optimizer)
    for i in preds:
        predicts = train_eval.decode_preds(i, DataSets.label_encoeder)
    scheduler.step(eval_loss)
    if eval_loss < min_loss:
        min_loss = eval_loss
        print(predicts)
    print('epoch {} \t eval_loss {} \t '.format(epoch,eval_loss))
    
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0,0,0], [0.01,0.01,0.01])])
testi = cv2.imread(r'C:/Users/tamiryuv/Desktop/Play/Cpatcha OCR/captcha_images_v2/2b827.png')
testi = transforms(testi)
testi.unsqueeze_(0)
outs = [0]
for i in range(5):
    trg_tensor = torch.LongTensor(outs).unsqueeze(0)
    Captcha_model.eval()
    with torch.no_grad():
        preds = Captcha_model(testi,trg_tensor,test = True)
        print(preds.shape)
        best_guss = preds[-1][0].argmax().item()
        print(best_guss)
        outs.append(best_guss)
