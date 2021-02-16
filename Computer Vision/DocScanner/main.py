# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:33:14 2020

@author: tamiryuv
"""
import numpy as np
import cv2
import utils

path = r'C:/Users/tamiryuv/Desktop/Play/OPENCV/Doc_Scanner/test2.jpeg'

image = cv2.imread(path)
im1 = cv2.resize(image,(600,600))
gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),1)
thresh = cv2.adaptiveThreshold(blur,255,0,1,11,2)


contuors,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contuors = sorted(contuors,key = lambda x: cv2.contourArea(x),reverse=True)#[:5]
corners = []
for i,con in enumerate(contuors):
    peri = cv2.arcLength(con, True)
    approx = cv2.approxPolyDP(con, 0.02 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our doc
    if len(approx) == 4:
        corners.append(approx)
max_area = 0
for idx,c in enumerate(corners):
    if cv2.contourArea(c) > max_area:
        max_area = cv2.contourArea(c)
        best = idx
    
           

best_conts = corners[best]    
    
pts1,pts2 = utils.rank(im1,best_conts)

matrix = cv2.getPerspectiveTransform(pts1,pts2)


output = cv2.warpPerspective(im1,matrix,(600,600))
output = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)

kernel = np.array( [[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]])

sharpened = cv2.filter2D(output, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
for x in range(4):
    cv2.circle(im1,(pts1[x][0],pts1[x][1]),5,(0,0,255))
cv2.imshow('original',im1)
cv2.imshow('inp',thresh)
#cv2.imshow('pl', output)
cv2.imshow('res', sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
