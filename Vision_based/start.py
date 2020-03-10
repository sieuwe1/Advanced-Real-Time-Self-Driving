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

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

from lines import *
from steering import *

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--video', type=str, default='', help='path to video', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
args = parser.parse_args()
print(args)
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

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

cap = cv2.VideoCapture(args.video)

framecount = 0
centerPoints = []

center =(0+int(512 / 2),0+int(256 / 2))

prevFrame = None

def filterConnectedComponents(pred):
    label_img, cc_num = ndimage.label(pred)
    sizes = ndimage.sum(pred, label_img, range(cc_num+1))
    #print(sizes)
    mask_size = sizes < 100000
    remove_pixel = mask_size[label_img]
    label_img[remove_pixel] = 0
    return label_img

def CalculateOpticalFlowValue(nextFrame):
    colorized = args.dataset_cls.colorize_mask(prevFrame)
    prevFrameColor = np.array(colorized.convert('RGB'))
    prevFrameGray = cv2.cvtColor(prevFrameColor,cv2.COLOR_BGR2GRAY)
    nextFrameGray = cv2.cvtColor(nextFrame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevFrameGray, nextFrameGray, None,0.5,3,15,3,5,1.2,0)
    return np.mean(flow)  
  

while True:

    beginTime = datetime.datetime.now()

    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img,(512,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    if framecount % 10 == 0:   
        img2 = img
        img_tensor = img_transform(img2)

        # predict
        with torch.no_grad():
            img2 = img_tensor.unsqueeze(0).cuda().cpu()
            pred = net(img2)

        pred = pred.cpu().numpy().squeeze()
        pred = np.argmax(pred, axis=0)
        
        #colorized1 = args.dataset_cls.colorize_mask(pred)
        #cv2.imshow("net OUT",np.array(colorized1.convert('RGB')))

        #apply connected component method to find largest connected object
        smooth_pred = ndimage.gaussian_filter(pred,3.0)       
        filtered_pred = filterConnectedComponents(smooth_pred)

        #show
        colorized = args.dataset_cls.colorize_mask(filtered_pred)
        filteredFrame = np.array(colorized.convert('RGB'))
        cv2.imshow("net OUT filtered",filteredFrame)

        centerPoints, img = getLines(filtered_pred, img)

        if len(centerPoints) > 5:
            img = getSteeringCommand(center, centerPoints, img)

        endTime = datetime.datetime.now()

        elapsedTime = endTime - beginTime
        if(elapsedTime.microseconds > 0.0):
            fps = round(1 / (elapsedTime.microseconds * 10**-6),2) 
            cv2.putText(img,"fps: " + str(fps),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4,cv2.LINE_AA)


        cv2.imshow("OUT", img)
   
    k = cv2.waitKey(1)
    if k == 27:
        break
    framecount += 1

cv2.imshow("last",img)
cv2.waitKey(0)


