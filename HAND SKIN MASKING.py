# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:50:05 2018

@author: Vishwas
"""
import numpy as np
import cv2 as cv
import os


cap = cv.VideoCapture(2)
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame=cv.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)
    
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    img=frame[170:370, 100:300]
    
    cv.rectangle(frame,(100,170),(300,370),(0,255,0),3)
    
    def nothing(x):
        pass
    #YCbCr conversion
    
    YCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    

    cv.namedWindow('image')
    # create trackbars for color change
    cv.createTrackbar('Y1','image',70,255,nothing)
    cv.createTrackbar('Cb1','image',85,255,nothing)
    cv.createTrackbar('Cr1','image',135,255,nothing)
    cv.createTrackbar('Y2','image',255,255,nothing)
    cv.createTrackbar('Cb2','image',135,255,nothing)
    cv.createTrackbar('Cr2','image',180,255,nothing)
    cv.createTrackbar('p','image',1,10,nothing)
    cv.createTrackbar('q','image',1,10,nothing)
    
    Y1 = cv.getTrackbarPos('Y1','image')
    Cb1 = cv.getTrackbarPos('Cb1','image')
    Cr1 = cv.getTrackbarPos('Cr1','image')
    Y2 = cv.getTrackbarPos('Y2','image')
    Cb2 = cv.getTrackbarPos('Cb2','image')
    Cr2 = cv.getTrackbarPos('Cr2','image')
    p = cv.getTrackbarPos('p','image')
    q = cv.getTrackbarPos('q','image')
    
    
    minYCB = np.array([Y1, Cr1, Cb1])
    maxYCB = np.array([Y2, Cr2, Cb2])

   # minYCB = np.array([80, 135, 85])
   # maxYCB = np.array([255, 180, 135])

    maskYCB = cv.inRange(YCbCr, minYCB, maxYCB)
    resultYCB = cv.bitwise_and(YCbCr, YCbCr, mask = maskYCB)
    
    erosion = cv.erode(maskYCB,kernel,iterations = p)
    
    dilation = cv.dilate(erosion,kernel,iterations = q)
    mask = cv.morphologyEx(maskYCB, cv.MORPH_OPEN, kernel)
    
    Gblur = cv.GaussianBlur(dilation,(5,5),100)
    #blur = cv.bilateralFilter(dilation,9,100,100)
    #Final = cv.resize(img,None,fx=2, fy=2, interpolation = CV_INTER_AREA)
   # cv.resize(Gblur,Final,dst_size(),10,10,);
   # cv.imshow('mask',mask)
    cv.imshow('dilation',dilation)
 #   cv.imshow('Final',Final)
    #cv.imshow('YCbCr',YCbCr)
  #  cv.imshow('resultYCB',resultYCB)
   # cv.imshow('blur',blur)
   # cv.imshow('maskYCB',maskYCB)
    cv.imshow('gblur',Gblur)
     
    cv.imshow('Img',img)
    # Display the resulting frame
    cv.imshow('frame',frame)
    path = 'D:/Images'

    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite(os.path.join(path,'frame%d.jpg') % count,Gblur)
        count += 1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()