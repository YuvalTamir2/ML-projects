# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:43:43 2021

@author: tamiryuv
"""
import os
import cv2
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceID(object):
    def __init__(self):
    
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20, keep_all = True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval() 
        
    def embedFace(self,image):
        #image = cv2.imread(image)
        embeds = []
        img_cropped_list, prob_list = self.mtcnn(image, return_prob=True) # get all faces
        if img_cropped_list is not None:
#            print(len(img_cropped_list))
            for i, prob in enumerate(prob_list):
                if prob>0.90:
#                    print('recognized')
                    embed = self.resnet(img_cropped_list[i].unsqueeze(0)).detach() # embed face by face
                    embeds.append(embed.detach())
            #outs = np.array(embeds)
            #outs = outs.reshape(outs.shape[0],-1)
            
            return embeds #outs
        
    def collat(self,x):
        return x[0]

    def train(self,root):
        data_set = datasets.ImageFolder(root = root)
        self.i2c = {i:c for c,i in data_set.class_to_idx.items()}
        loader = DataLoader(dataset=data_set, collate_fn=self.collat)
        
        embedding = []
        names = []
        
        for image,idx in loader:
            emb = self.embedFace(image)
            embedding.append(emb)
            names.append(self.i2c[idx])
        #print(embedding)
        trained_data = (embedding,names)
        torch.save(trained_data,'trained_data.pt')
        return 'finished training...\ndata saved in {}'.format(os.getcwd())
    
    
    
    
    
    def live_stream(self):
        
        data = torch.load('trained_data.pt')
        embeds = data[0]
        names = data[1]
        #print(embeds,names)
        cap = cv2.VideoCapture(0) 

        while True:
            booli, frame = cap.read()
            if not booli:
                print("camera problems..")
                break
                
            #img = Image.fromarray(frame)
            img_cropped_list, prob_list = self.mtcnn(frame, return_prob=True) 
            
            if img_cropped_list is not None:
                boxes, _ = self.mtcnn.detect(frame)
                        
                for i, prob in enumerate(prob_list):
                    if prob>0.90:
                        emb = self.resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                        
                        dist_list = [] # list of matched distances, minimum distance is used to identify the person
                        
                        for idx, emb_db in enumerate(embeds):
                            dist = torch.dist(emb, emb_db[0]).item()
                            dist_list.append(dist)
                        min_dist = min(dist_list) # get minumum dist value
                        min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                        name = names[min_dist_idx] # get name corespondes to min_dist value
                        box = boxes[i] 
                        
                        if min_dist<1:
                            frame = cv2.putText(frame, name+' '+str(np.round(min_dist,3)), (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (115,64,255),1)
                        
                        frame = cv2.rectangle(frame, (box[0],box[1]) , (box[2],box[3]), (255,0,0), 2)
        
            cv2.imshow("Output", frame)
                
            
            k = cv2.waitKey(1)
            if k%256==27: # ESC
                print('Esc pressed, closing...')
                break
        cap.release()
        cv2.destroyAllWindows()
        
        
    
    
    
        
        
        
        
