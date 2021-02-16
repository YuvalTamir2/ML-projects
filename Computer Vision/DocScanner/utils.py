# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:13:54 2020

@author: tamiryuv
"""

import cv2
import numpy as np

def rank(original_image,contours):
    w,h = original_image.shape[:2]
    new_conts = np.empty((4,2))
    indices_used = []
    sums = []
    for idx,cont in enumerate(contours):
        sums.append(cont.sum())
    new_conts[0] = contours[sums.index(min(sums))]
    new_conts[3] = contours[sums.index(max(sums))]
    indices_used.extend([sums.index(min(sums)),sums.index(max(sums))])
    min_y = 1000
    for idx,cont in enumerate(contours):
        if idx not in indices_used:
            x,y = cont[0][0],cont[0][1]
            if y < min_y:
                min_y = y
                best_idx = idx
            second_point = best_idx
    indices_used.append(best_idx)
    new_conts[1] = contours[second_point]
    last_idx = [i for i,j in enumerate(contours) if i not in indices_used][0]
    new_conts[2] = contours[last_idx]
    
    
    pts1 = np.float32(new_conts)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    
    return pts1,pts2

                                           
    
    
    