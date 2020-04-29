import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
import skimage
from skimage import morphology
import datetime
import json
import freenect
import math
import time
from matplotlib import pyplot as plt

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

import donkeycar as dk

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--video', type=str, default='', help='path to video', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_cv0_seresnext50_nosdcaug.pth', help='pre-trained checkpoint')
parser.add_argument('--arch', type=str, default='network.deepv3.DeepSRNX50V3PlusD_m1', help='network architecture used for inference')
args = parser.parse_args()
print(args)
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

#load donkeyCar config
cfg = dk.load_config()

print('starting segmentation network....')

# get net
args.dataset_cls = cityscapes
net = network.get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
print('Net restored.')

# get data
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

#Blob detector
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 1000

# Filter by Area.
params.filterByArea = True
params.minArea = 0.01

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.02

detector = cv2.SimpleBlobDetector_create(params)

def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

#load model 
def loadModel(model_path):
    #from donkeyCar file complete.py
    kl = dk.utils.get_model_by_type(cfg.DEFAULT_MODEL_TYPE, cfg)
    model_reload_cb = None

    if '.h5' in model_path or '.uff' in model_path or 'tflite' in model_path or '.pkl' in model_path:
        #when we have a .h5 extension
        #load everything from the model file
         kl.load(model_path)
    
    return kl  

#driveModel = loadModel('LinearRunFixed.h5')
driveModel = loadModel('TrainedModel.h5')

def filterBlobs(image,size):
    
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Detect blobs.
    keypoints = detector.detect(imageGray)
    
    for x in range(1,len(keypoints)):
        image=cv2.circle(image, (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])), radius=np.int(keypoints[x].size), color=(0,255,0), thickness=-1)

    return image

def filterContours(image,size):
    edged = cv2.Canny(image, 175, 200)

    cv2.imshow("canny", edged)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, contours, -1, (0,0,255), 100)


    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        #if area < size:
            #cv2.drawContours(image,[contour],0,255,-1)
        cv2.fillPoly(image, pts =[contour], color=(0,0,255), maxLevel=2)
        #cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)
        #hull = cv2.convexHull(cnt)

    return image

def adjust_gamma(image, gamma):
    image = cv2.resize(image,(512,256))
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def draw_model_prediction(pilot_angle, pilot_throttle, img):
    '''
    query the model for it's prediction, draw the predictions
    as a blue line on the image

    Taken from https://github.com/autorope/donkeycar/blob/bd854d3de6b109d9ae711dba271305f4b4c0c55d/donkeycar/management/makemovie.py
    '''

    height, width, _ = img.shape

    length = height
    a2 = pilot_angle * 45.0
    l2 = pilot_throttle * length

    mid = width // 2 - 1

    p2 = tuple((mid + 2, height - 1))
    p22 = tuple((int(p2[0] + l2 * math.cos((a2 + 270.0) * (math.pi / 180.0))),
                    int(p2[1] + l2 * math.sin((a2 + 270.0) * (math.pi / 180.0)))))

    # user is green, pilot is blue
    cv2.line(img, p2, p22, (0, 0, 255), 2)

    return img

cap = cv2.VideoCapture("OcciRunGood.avi")
framecount = 0

while True:

    ret, img = cap.read()
    if not ret:
        break

    cv2.imshow("Input_image", img)
    
    img = adjust_gamma(img,1)

    #cv2.imshow("gamma", img)

    if framecount % 1 == 0:   
        img2 = img
        img_tensor = img_transform(img2)

        # predict
        with torch.no_grad():
            img2 = img_tensor.unsqueeze(0).cuda().cpu()
            pred = net(img2)

        pred = pred.cpu().numpy().squeeze()
        pred = np.argmax(pred, axis=0)
            
        #apply connected component method to find largest connected object. Use this to only train the road to the network
        #smooth_pred = ndimage.gaussian_filter(pred,3.0) 

        colorized = args.dataset_cls.colorize_mask(pred)
        img = np.array(colorized.convert('RGB'))

        #cv2.imshow("before", img)

        kernel = np.ones((15,15),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        median = cv2.medianBlur(img, 17)

        cv2.imshow("after", median)
        
        #img = filterContours(median,10000000)

        #cv2.imshow("after", img)


        #filtered_pred = filterContours(smooth_pred)
        #filtered_pred = filterConnectedComponents(smooth_pred)
        #print(type(filtered_pred))     

        #filtered_pred = cv2.normalize(filtered_pred, None, 0, 255, cv2.NORM_MINMAX)

        #print(filtered_pred)

        #cv2.imshow('Segmenteted_image',filtered_pred)

        #predict control 
        pred_img = median.astype(np.float32) / 255.0
        steeringValue , throttleValue = driveModel.run(pred_img)
        steeringValue -= 0.23
        pred_img = draw_model_prediction(steeringValue, throttleValue, pred_img)
        print(steeringValue, throttleValue)
        cv2.imshow('prediction',pred_img)


    k = cv2.waitKey(1)
    if k == 27:
        break
    framecount += 1
