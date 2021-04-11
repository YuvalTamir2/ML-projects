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
import os
import glob
import random
# train_loader = DataLoader(DataSets.train_dataset, batch_size=32, shuffle=True)
# test_loader =DataLoader(DataSets.test_dataset, batch_size=32, shuffle = False)

EPOCHES = 100
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
        #print(predicts)
    print('epoch {} \t eval_loss {} \t '.format(epoch,eval_loss))


def predict(im_path):
    global Captcha_model
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0,0,0], [0.01,0.01,0.01])])
    testi = cv2.imread(im_path)
    testi = transforms(testi)
    testi.unsqueeze_(0)
    outs = [0]
    for i in range(5):
        trg_tensor = torch.LongTensor(outs).unsqueeze(0)
        Captcha_model.eval()
        with torch.no_grad():
            preds = Captcha_model(testi,trg_tensor,test = True)
            #print(preds.shape)
            best_guss = preds[-1][0].argmax().item()
            #print(best_guss)
            outs.append(best_guss)
    outs = list(map(lambda x: x - 1,outs[1:]))
    preds = []
    for token in outs:
        if token != -1:
            ar = DataSets.label_encoeder.inverse_transform([token]).item()
            preds.append(ar)
        else:
            break
        
    
    return ''.join(preds)

o = predict(r'C:/Users/tamiryuv/Desktop/Play/Cpatcha OCR/captcha_images_v2/3043-Captcha-smwm.svg.png')
print(o)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def plot_examples(list_of_images):
    # 19 images, if wanna change,
    # change the num of columns and rows so that col*rows - 1 = num images
    fig = plt.figure(figsize=(9, 13))
    ax = []
    columns = 4
    rows = 5
    for i,img in enumerate(list_of_images):
        o = predict(img)
        image = mpimg.imread(img)
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title("pred:"+str(o))  # set title
        plt.imshow(image, alpha=0.25)
    plt.show()
    
data_file_name = os.listdir()[0]
images_path = glob.glob(os.path.join(data_file_name,'*.png'))

list_of_images = random.sample(images_path,19)
plot_examples(list_of_images)



















    