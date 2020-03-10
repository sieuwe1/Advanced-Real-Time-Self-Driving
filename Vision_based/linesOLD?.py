import numpy as np
import cv2
import math
import pandas as pd
from collections import deque
#array = np.loadtxt("/home/sieuwe/Desktop/Projects/semantic-segmentation-master/test.txt", delimiter=',')


#img = cv2.imread('test.png',1)
#img = cv2.resize(img, (639, 185))

maxDistance = 100
minRoadWidth = 30
prevRoadWidthList = deque(maxlen = 10) 
prevRoadWidthList.append(100)

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

    #get length of shortest side 
    length = 0
    if(len(lineListleft) > len(lineListright)):
        length = len(lineListright) - 1
    else: 
        length = len(lineListleft) - 1
        
    centerPointsY = []
    centerPointsX = []

    #filter bad points and draw center/side lines
    leftpoint = (0,0)
    rightpoint = (0,0)

    roadWidthStartIndex = length - round(length / 3)
    roadWidthList = []
    maxRoadWidth = round(sum(prevRoadWidthList) / len(prevRoadWidthList) *1.05)
    
   # print(maxRoadWidth)
   # global prevRoadWidthList
    
    prevRoadWidth = maxRoadWidth / 1.05
    roadWidth = 0

    for p in range(length):
        print(roadWidth)
        print("prev " + str(prevRoadWidth))
        
        if calculateDistance(lineListleft[p][0], lineListleft[p][1], lineListleft[p+1][0], lineListleft[p+1][1]) < maxDistance:
            leftpoint = lineListleft[p+1]
            cv2.line(img, lineListleft[p], lineListleft[p+1], [0, 255, 0], 3)
            roadWidth = abs(leftpoint[0] - rightpoint[0])
            prevRoadWidth = roadWidth
        
        else:
            roadWidth = prevRoadWidth

        if calculateDistance(lineListright[p][0], lineListright[p][1], lineListright[p+1][0], lineListright[p+1][1]) < maxDistance:
            rightpoint = lineListright[p+1]
            cv2.line(img, lineListright[p], lineListright[p+1], [255, 0, 0], 3)
            roadWidth = abs(leftpoint[0] - rightpoint[0])
            prevRoadWidth = roadWidth

        else:
            roadWidth = prevRoadWidth

        print("min " + str(minRoadWidth))
        print("max " + str(maxRoadWidth))

        if roadWidth > minRoadWidth and roadWidth < maxRoadWidth:
            centerPointsX.append(leftpoint[0] + round(roadWidth / 2))
            centerPointsY.append(leftpoint[1])
        
        else:
            if roadWidth > maxRoadWidth:
                print("Possible side exit?")
                #print("preWidth" + str(prevRoadWidth))
                #print("width" + str(roadWidth))
                #cv2.imwrite("lol.jpg", img)
                #a = (lineListleft[p])
                #b = (lineListright[p])
                #center = (centerPointsX[len(centerPointsX) - 1], centerPointsY[len(centerPointsX) - 1])
                
                #if(calculateDistance(a[0],a[1],center[0],center[1]) > calculateDistance(b[0],b[1],center[0],center[1]) ):
                #    print("left side exit?")
                #cv2.line(img,(lineListright[p][0] - round(maxRoadWidth / 1.15) ,lineListright[p][1]), (lineListright[p+1][0] - round(maxRoadWidth / 1.15),lineListright[p][1]), [255,255,0],3)

                #else:
                #    print("right exit?")
                #cv2.line(img,(lineListleft[p][0] + round(maxRoadWidth / 1.05) ,lineListleft[p][1]), (lineListleft[p+1][0] + round(maxRoadWidth / 1.05),lineListleft[p][1]), [255,255,0],3)
                centerPointsX.append(centerPointsX[len(centerPointsX) - 1])
                centerPointsY.append(centerPointsY[len(centerPointsY) - 1])

        if p > roadWidthStartIndex:
            roadWidthList.append(roadWidth)
    
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

