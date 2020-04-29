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

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

from vehicleSerial import *
from fileWriter import *

parser = argparse.ArgumentParser(description='drive training data')
parser.add_argument('--savevideo', type=str, default='', help='save incoming video')
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_cv0_seresnext50_nosdcaug.pth', help='pre-trained checkpoint')
parser.add_argument('--arch', type=str, default='network.deepv3.DeepSRNX50V3PlusD_m1', help='network architecture used for inference')
args = parser.parse_args()
print(args)
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
beginTime = datetime.date
torch.cuda.empty_cache()

setupWriter(False)

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

#cap = cv2.VideoCapture(args.video)

framecount = 0

prevFrame = None


def draw_user_angle(pilot_angle, pilot_throttle, img):
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


def filterConnectedComponents(pred):
    label_img, cc_num = ndimage.label(pred)
    sizes = ndimage.sum(pred, label_img, range(cc_num+1))
    #print(sizes)
    mask_size = sizes < 50
    remove_pixel = mask_size[label_img]
    label_img[remove_pixel] = 0
    return label_img

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

#connect to autonomous vehicle
connectionResult = connect('/dev/ttyUSB0')

while connectionResult != "succes":
    print("arduino failed to connect but trying again...")
    connectionResult = connect('/dev/ttyUSB0')

print("arduino connected!")

if args.savevideo:
    out = cv2.VideoWriter("OcciRunBad" + ".avi", cv2.VideoWriter_fourcc('M','J','P','G'),10,(512,256))

cap = cv2.VideoCapture(0)

while True:

    #img = get_video()
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img,(512,256))
    cv2.imshow("IN", img)
    
    if args.savevideo:
        out.write(img)

    if framecount % 5 == 0:   
        #reading steering and throttle value as fast as possible to make sure there is no delay
        
        steeringValue, throttleValue = readValue()

        if steeringValue != -1 or steeringValue != 1:
            img2 = img
            img_tensor = img_transform(img2)

            # predict
            with torch.no_grad():
                img2 = img_tensor.unsqueeze(0).cuda().cpu()
                pred = net(img2)

            pred = pred.cpu().numpy().squeeze()
            pred = np.argmax(pred, axis=0)
 
            colorized = args.dataset_cls.colorize_mask(pred)
            img = np.array(colorized.convert('RGB'))
            kernel = np.ones((15,15),np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            median = cv2.medianBlur(img, 17)

            #write data to disk for training
            writeTrainData(steeringValue,throttleValue,median,int(framecount/5))

            #endTime = datetime.datetime.now()

            #elapsedTime = endTime - beginTime
            #if(elapsedTime.microseconds > 0.0):
            #    fps = round(1 / (elapsedTime.microseconds * 10**-6),2) 
                #cv2.putText(img,"fps: " + str(fps),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4,cv2.LINE_AA)

            #print(str(steeringValue) + ", " + str(throttleValue))
            median = draw_user_angle(steeringValue,throttleValue,median)

            cv2.imshow("OUT", median) 
    
    k = cv2.waitKey(1)
    if k == 27:
        break
    framecount += 1
