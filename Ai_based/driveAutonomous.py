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
import threading

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

from vehicleSerial import *
from control import *

import donkeycar as dk
from donkeycar.utils import *


parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--video', type=str, default='', help='path to video', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
args = parser.parse_args()
print(args)
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

#load donkeyCar config
cfg = dk.load_config()

#create data folder
fileName = input("Press enter to start vehicle")

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

cap = cv2.VideoCapture(args.video)

framecount = 0
driveModel = None

def filterConnectedComponents(pred):
    label_img, cc_num = ndimage.label(pred)
    sizes = ndimage.sum(pred, label_img, range(cc_num+1))
    #print(sizes)
    mask_size = sizes < 100000
    remove_pixel = mask_size[label_img]
    label_img[remove_pixel] = 0
    return label_img

#brigthen image because xbox kinect rgb cam is pretty shitty
def adjust_gamma(image, gamma):
    image = cv2.resize(image,(512,256))
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

#load model 

def load_model(kl, model_path):
    start = time.time()
    print('loading model', model_path)
    kl.load(model_path)
    print('finished loading in %s sec.' % (str(time.time() - start)) )

def loadModel(model_path):
    #from donkeyCar file complete.py
    kl = dk.utils.get_model_by_type(cfg.DEFAULT_MODEL_TYPE, cfg)
    model_reload_cb = None

    if '.h5' in model_path or '.uff' in model_path or 'tflite' in model_path or '.pkl' in model_path:
        #when we have a .h5 extension
        #load everything from the model file
        load_model(kl, model_path)
    
    return kl  
    
driveModel = loadModel('GoogleAdam.h5')

#connect to autonomous vehicle
connectionResult = connect('/dev/ttyUSB0')

while connectionResult != "succes":
    print("arduino failed to connect but trying again...")
    connectionResult = connect('/dev/ttyUSB0')

print("arduino connected!")

time.sleep(3)

#controlThread = threading.Thread(target=main)
#controlThread.start()

while True:
    beginTime = datetime.datetime.now()

    ret, img = cap.read()
    if not ret:
        print("video not found!")
        break

    img = adjust_gamma(img,2)
    cv2.imshow("IN", img)

    if framecount % 10 == 0:   
        
        steeringValue = None
        throttleValue = None

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
        #pred = filterConnectedComponents(smooth_pred)

        #show
        colorized = args.dataset_cls.colorize_mask(pred)
        img = np.array(colorized.convert('RGB'))
        
        steeringValue, throttleValue= driveModel.run(img)

        setDesiredSteeringAngle(steeringValue / 100)
        setDesiredThrottle(throttleValue/ 100)

        endTime = datetime.datetime.now()

        elapsedTime = endTime - beginTime
    
        if(elapsedTime.microseconds > 0.0):
            fps = round(1 / (elapsedTime.microseconds * 10**-6),2) 
            #cv2.putText(img,"fps: " + str(fps),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4,cv2.LINE_AA)
            #print("steeringValue > " + str(steeringValue) + " throttleValue > " + str(throttleValue) + " fps: " + str(fps))

        cv2.imshow("OUT", img) 
   
    k = cv2.waitKey(1)
    if k == 27:
        break
    framecount += 1
