# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:50:14 2021

@author: tamiryuv
"""
import os
import torch
import numpy as np
import cv2
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import matplotlib.pyplot as plt

data_file_name = os.listdir()[0]
images_path = glob.glob(os.path.join(data_file_name,'*.png'))[:500]
label_encoeder = LabelEncoder()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_targets(images_path):
    targets = []
    global label_encoeder
    for image in images_path:
        im = image.split('\\')[-1].split('.')[0]
        target = [i for i in im]
        targets.append(target)
    tar_flat = np.array(targets).ravel()
    # print(tar_flat.shape)
    label_encoeder.fit(tar_flat)
    taregt_clean = np.array([label_encoeder.transform(x) for x in targets])
    return taregt_clean + 1
    
targets = get_targets(images_path)
targets = np.insert(targets, 0, values=0, axis=1)
targets = np.append(targets, np.zeros((targets.shape[0],1)), axis=1)
NUM_CHARS = len(np.unique(targets.ravel()))


X_train,X_test, y_train, y_test = train_test_split(images_path,targets, test_size = 0.4)


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0,0,0], [0.01,0.01,0.01])])

class CaptchaDataset(Dataset):
    
    def __init__(self,images, targets, transform = None):
        
        self.images = images
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image = cv2.imread(self.images[index])
#        image = cv2.resize(image, (300,75))
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
            
        target = self.targets[index]
        
        return image, target
    
train_dataset = CaptchaDataset(X_train,y_train,transforms)
test_dataset = CaptchaDataset(X_train,y_train,transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader =DataLoader(test_dataset, batch_size=32, shuffle = False)

