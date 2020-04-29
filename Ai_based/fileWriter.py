#handels all writing to disk
import os
import cv2
import datetime
import json
import math

dataName = None 
inImageWriter = None 
outImageWriter = None 
debugFile = None
data_dir = None

def setupWriter(debug):
    global dataName, inImageWriter, outImageWriter, debugFile, data_dir

    dataName = input("please type name of this run. This will be the name of the Data folder> ")

    current_directory = os.getcwd()
    data_dir = os.path.join(current_directory, dataName)
    try:
        os.mkdir(data_dir)
        print("Directory " , data_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , data_dir ,  " already exists")

    if debug:
        debug_dir = os.path.join(current_directory, "DEBUG_" + dataName)
        try:
            os.mkdir(debug_dir)
            print("Directory " , debug_dir ,  " Created ") 
        except FileExistsError:
            print("Directory " , debug_dir ,  " already exists")
        
        inImagePath = os.path.join(debug_dir, "InImage.avi")
        inImageWriter = cv2.VideoWriter(inImagePath,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,(512,256))
        outImagePath = os.path.join(debug_dir, "OutImage.avi")
        outImageWriter = cv2.VideoWriter(outImagePath,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,(512,256))

        debugFilePath = os.path.join(debug_dir, "DebugMessage.txt")
        debugFile = open(debugFilePath, "w")

#Train data 
def writeTrainData(angle, throttle, image, framecount):

    camName = str(framecount) + '_cam-image_array_.jpg'
    camPath = os.path.join(data_dir, camName)
    cv2.imwrite(camPath, image) 
    json_data = {"user/angle": angle, "cam/image_array": camName, "user/throttle": throttle, "user/mode": "user"}
    
    jsonName = "record_" + str(framecount) + '.json'
    jsonPath = os.path.join(data_dir, jsonName)
    with open(jsonPath, "w") as write_file:
        json.dump(json_data, write_file)
    

#Debugging messages
def writeInDebugImage(InImage):
    inImageWriter.write(InImage)

def writeOutDebugImage(OutImage):
    outImageWriter.write(OutImage)

def writeAngleAndThrottle(angle, throttle, framecount):
    debugFile.write(str(framecount) + " angle: " + str(angle) + " throttle: " + str(throttle))
    debugFile.write("\n")

def writeControlMessage(message):
    debugFile.write(message)
    debugFile.write("\n")