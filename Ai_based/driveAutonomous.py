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
import freenect

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

#These offsets can be used when sensors where not placed correclty during training 
steerOffset = -0.22; #Is added to predicted steer angle
throttleOffset = 0; # is added to predicted throttle value

parser = argparse.ArgumentParser(description='Drive autonomous')
parser.add_argument('--debug', type=bool, default='False', help='enable debug messages')
parser.add_argument('--video', type=str, default='.', help='path to video')
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_cv0_seresnext50_nosdcaug.pth', help='pre-trained checkpoint')
parser.add_argument('--arch', type=str, default='network.deepv3.DeepSRNX50V3PlusD_m1', help='network architecture used for inference')
args = parser.parse_args()
print(args)
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

#load donkeyCar config
cfg = dk.load_config()

#setup file writer 
setupWriter(args.debug)

print('starting segmentation network....')
# get net
args.dataset_cls = cityscapes
net = network.get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
print('Net restored.')

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

framecount = 0
driveModel = None

cap = cv2.VideoCapture(0)

#help funtions 
def filterConnectedComponents(pred):
    label_img, cc_num = ndimage.label(pred)
    sizes = ndimage.sum(pred, label_img, range(cc_num+1))
    #print(sizes)
    mask_size = sizes < 100000
    remove_pixel = mask_size[label_img]
    label_img[remove_pixel] = 0
    return label_img


def draw_model_prediction(pilot_angle, pilot_throttle, img, color):
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
    cv2.line(img, p2, p22, color, 2)

    return img

#brigthen image because xbox kinect rgb cam is pretty shitty
def adjust_gamma(image, gamma):
    image = cv2.resize(image,(512,256))
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

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

#connect to autonomous vehicle
connectionResult = connect('/dev/ttyUSB0')

while connectionResult != "succes":
    print("arduino failed to connect but trying again...")
    connectionResult = connect('/dev/ttyUSB0')

print("arduino connected!")
input("Press enter to start vehicle")


#small workaround may fix later
debugString = "False"
if args.debug:
    debugString = "True"
    

controlThread = threading.Thread(target=main, args=(debugString,))
controlThread.start()

while True:
    beginTime = datetime.datetime.now()

    #img = get_video()
    ret, img = cap.read()
    img = cv2.resize(img,(512,256))

    cv2.putText(img,"Frame: " + str(framecount),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4,cv2.LINE_AA)
    cv2.imshow("IN", img)

    if args.debug:
        writeInDebugImage(img)

    if framecount % 5 == 0:   
        
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
        
        kernel = np.ones((15,15),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        median = cv2.medianBlur(img, 17)

        pred_img = median.astype(np.float32) / 255.0

        steeringValue, throttleValue = driveModel.run(pred_img)

        steeringValue = steeringValue + steerOffset
        throttleValue = throttleValue + throttleOffset

        img = draw_model_prediction(steeringValue, throttleValue, img, (0, 0, 255))

        setDesiredSteeringAngle(steeringValue)
        setDesiredThrottle(throttleValue)

        currentSteeringValue, currentThrottleValue = getCurrentStatus()

        img = draw_model_prediction(currentSteeringValue, currentThrottleValue, img, (0, 255, 255))

        endTime = datetime.datetime.now()

        elapsedTime = endTime - beginTime
    
        if(elapsedTime.microseconds > 0.0):
            fps = round(1 / (elapsedTime.microseconds * 10**-6),2) 
            print("steeringValue > " + str(steeringValue) + " throttleValue > " + str(throttleValue) + " fps: " + str(fps))

        cv2.imshow("OUT", img) 
   
        if args.debug:
            writeOutDebugImage(img)
            writeAngleAndThrottle(steeringValue,throttleValue,framecount)

    k = cv2.waitKey(1)
    if k == 27:
        break
    framecount += 1
