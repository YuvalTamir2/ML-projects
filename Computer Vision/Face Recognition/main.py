# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:54:37 2021

@author: tamiryuv
"""


from FaceNet import FaceID
import os


def main_fn(test = False):

    face_id = FaceID()
    root = os.path.join(os.getcwd(),'images')
    face_id.train(root)
    if test:
        face_id.live_stream()
        
        
if __name__ == "__main__":
    main_fn(True)




