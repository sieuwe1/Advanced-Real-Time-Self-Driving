import numpy as np
import cv2
import math
import pandas as pd
from collections import deque
#array = np.loadtxt("/home/sieuwe/Desktop/Projects/semantic-segmentation-master/test.txt", delimiter=',')


#img = cv2.imread('test.png',1)
#img = cv2.resize(img, (639, 185))

sideExitDetectionRate = 50 #smaller value the faster it detects an exit
maxDistance = 50 #the maximum distance two points from iether the left or right side can be from each other.
minRoadWidth = 30 #the minimum size the road needs to have to be a valid road in pixels
prevRoadWidthList = deque(maxlen = 60) 
prevRoadWidthList.append(600)

def getLines(array,img):

    #DEBUG
    #cv2.imwrite("inimage.jpg", img)
    #np.set_printoptions(suppress=True)
    #np.savetxt('inarray.txt', array, delimiter=',')

    lineListleft = []
    lineListright = []
    preVal = 99
    preIndex = (0,0)
    
    #get sides of the road
    for index, x in np.ndenumerate(array.astype(np.float)):

        if(preVal > 0 and x == 0):
            if index[1] > 0:
                lineListleft.append((index[1],index[0])) 
            #print("L: " + str((index[1],index[0])))

        if(preVal == 0 and x > 0):
            if index[1] > 0:
                lineListright.append((preIndex[1],preIndex[0]))
            #print("R: " + str((preIndex[1],preIndex[0])))

        preIndex = index
        preVal = x

    #lineListleft.reverse() 
    #lineListright.reverse()

    lineListleftLength = len(lineListleft) - 2
    lineListrightLength = len(lineListright) - 2

    #get length of shortest side 
    length = 0
    if(lineListleftLength > lineListrightLength):
        length = lineListleftLength
    else: 
        length = lineListrightLength
        
    centerPointsY = []
    centerPointsX = []

    #filter bad points and draw center/side lines
    leftpoint = (0,0)
    rightpoint = (0,0)

    roadWidthStartIndex = length - round(length / 3)
    roadWidthList = []
    maxRoadWidth = round(sum(prevRoadWidthList) / len(prevRoadWidthList) *1.10)
    
   # print(maxRoadWidth)
   # global prevRoadWidthList
    
    prevRoadWidth = maxRoadWidth / 1.10
    roadWidth = 0

    for p in range(length):

        exitLocation = ""
        print(prevRoadWidth)
        
        if p > lineListleftLength:
            if lineListrightLength - lineListleftLength > sideExitDetectionRate:
                print("possible exit left?") #Uses previus roadwidth known good side to predict centerpoint
                centerPointsX.append(lineListright[p][0] - round(prevRoadWidth / 2))
                centerPointsY.append(lineListright[p][1])
                exitLocation = "left"
        
        else:
            if calculateDistance(lineListleft[p][0], lineListleft[p][1], lineListleft[p+1][0], lineListleft[p+1][1]) < maxDistance:
                leftpoint = lineListleft[p+1]
                cv2.line(img, lineListleft[p], leftpoint, [0, 255, 0], 3)
            else:
                leftpoint = 0

        if p > lineListrightLength:
            if lineListleftLength - lineListrightLength > sideExitDetectionRate:
                print("possible exit right?") #Uses previus roadwidth known good side to predict centerpoint
                if roadWidth > minRoadWidth and roadWidth < maxRoadWidth:
                    centerPointsX.append(lineListleft[p][0] + round(prevRoadWidth / 2))
                    centerPointsY.append(lineListleft[p][1])
                exitLocation = "right"

        else:
            if calculateDistance(lineListright[p][0], lineListright[p][1], lineListright[p+1][0], lineListright[p+1][1]) < maxDistance:
                rightpoint = lineListright[p+1]
                cv2.line(img, lineListright[p], rightpoint, [255, 0, 0], 3)
            else:
                rightpoint = 0

        if leftpoint != 0 and rightpoint != 0:
            roadWidth = abs(rightpoint[0] - leftpoint[0])
            
            if roadWidth > minRoadWidth and roadWidth < maxRoadWidth:
                if exitLocation == "left":
                    centerPointsX.append(rightpoint[0] - round(roadWidth / 2))
                    centerPointsY.append(rightpoint[1])
                else:
                    centerPointsX.append(leftpoint[0] + round(roadWidth / 2))
                    centerPointsY.append(leftpoint[1])

                prevRoadWidth = roadWidth

                if p > roadWidthStartIndex:
                    roadWidthList.append(roadWidth)

            else:
                if exitLocation == "right":
                    centerPointsX.append(rightpoint[0] - round(prevRoadWidth / 2))
                    centerPointsY.append(rightpoint[1])
                else:
                    centerPointsX.append(leftpoint[0] + round(prevRoadWidth / 2))
                    centerPointsY.append(leftpoint[1])
                print("error width " + str(roadWidth))

        
        print("min " + str(minRoadWidth))
        print("max " + str(maxRoadWidth))
    
    if len(roadWidthList) > 0:
        prevRoadWidthList.append(sum(roadWidthList) / len(roadWidthList))

    #moving average filter on centerline
    filteredCenterPointsX = []
    if len(centerPointsX) > 10:
        filteredCenterPointsX = movingaverage(centerPointsX,15)

    finalCenterPointsList = []

    for index in range(len(filteredCenterPointsX) - 1):
        cv2.line(img, (int(filteredCenterPointsX[index]), centerPointsY[index]), (int(filteredCenterPointsX[index + 1]), centerPointsY[index + 1]), [0, 0, 255], 3)
        finalCenterPointsList.append((int(filteredCenterPointsX[index]),centerPointsY[index]))

    
    return (finalCenterPointsList, img)

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

