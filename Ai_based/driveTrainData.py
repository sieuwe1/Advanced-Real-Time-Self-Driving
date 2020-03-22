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

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

from vehicleSerial import *

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--video', type=str, default='', help='path to video', required=True)
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
args = parser.parse_args()
print(args)
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

#create data folder
fileName = input("please type name of this run. This will be the name of the Data folder> ")

current_directory = os.getcwd()
data_dir = os.path.join(current_directory, fileName)
try:
    os.mkdir(data_dir)
    print("Directory " , data_dir ,  " Created ") 
except FileExistsError:
    print("Directory " , data_dir ,  " already exists")

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
cap = cv2.VideoCapture(0)

framecount = 0

prevFrame = None

def adjust_gamma(image, gamma):
    image = cv2.resize(image,(512,256))
    #invGamma = 1.0 / gamma
    #table = np.array([((i / 255.0) ** invGamma) * 255
    #for i in np.arange(0, 256)]).astype("uint8")
    #return cv2.LUT(image, table)
    return image


def filterConnectedComponents(pred):
    label_img, cc_num = ndimage.label(pred)
    sizes = ndimage.sum(pred, label_img, range(cc_num+1))
    #print(sizes)
    mask_size = sizes < 100000
    remove_pixel = mask_size[label_img]
    label_img[remove_pixel] = 0
    return label_img

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def writeData(angle, throttle, image):

    camName = str(framecount) + '_cam-image_array_.jpg'
    camPath = os.path.join(data_dir, camName)
    cv2.imwrite(camPath, image) 
    json_data = {"user/angle": angle, "cam/image_array": camName, "user/throttle": throttle, "user/mode": "user", "timestamp": int(time.time())}
    
    jsonName = "record_" + str(framecount) + '.json'
    jsonPath = os.path.join(data_dir, jsonName)
    with open(jsonPath, "w") as write_file:
        json.dump(json_data, write_file)

#connect to autonomous vehicle
connectionResult = connect('/dev/ttyUSB0')

while connectionResult != "succes":
    print("arduino failed to connect but trying again...")
    connectionResult = connect('/dev/ttyUSB0')

print("arduino connected!")

while True:

    beginTime = datetime.datetime.now()

    ret, img = cap.read()
    if not ret:
        break

    img = adjust_gamma(img,2)
    cv2.imshow("IN", img)

    if framecount % 10 == 0:   
        #reading steering and throttle value as fast as possible to make sure there is no delay
        
        steeringValue = None
        throttleValue = None

        result = readValue()
        if result == "error":
            print("!!!CONNECTION ERROR PRESS STOP EMEDIATLY!!!")
            break
        
        else: 
            steeringValue = result[0]
            throttleValue = result[1]

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

        #write data to disk for training
        writeData(steeringValue,throttleValue,img)

        endTime = datetime.datetime.now()

        elapsedTime = endTime - beginTime
        if(elapsedTime.microseconds > 0.0):
            fps = round(1 / (elapsedTime.microseconds * 10**-6),2) 
            #cv2.putText(img,"fps: " + str(fps),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4,cv2.LINE_AA)
            print("fps: " + str(fps))

        print(str(steeringValue) + ", " + str(throttleValue))
        cv2.imshow("OUT", img) 
   
    k = cv2.waitKey(1)
    if k == 27:
        break
    framecount += 1
