import numpy as np
import cv2
import math
import pandas as pd
#array = np.loadtxt("/home/sieuwe/Desktop/Projects/semantic-segmentation-master/test.txt", delimiter=',')


#img = cv2.imread('test.png',1)
#img = cv2.resize(img, (639, 185))

maxDistance = 30
maxVisionDistance = 10
minRoadWidth = 30

def getLines(array,img):

    cv2.imwrite("inimage.jpg", img)
    np.set_printoptions(suppress=True)
    np.savetxt('inarray.txt', array, delimiter=',')

    lineListleft = []
    lineListright = []
    totalCount = 0
    preVal = 99

    #get sides of the road
    for index, x in np.ndenumerate(array.astype(np.float)):
        
        if(preVal > 0 and x == 0):
            lineListleft.append((index[1],index[0])) 
        
        elif(preVal == 0 and x > 0):
            lineListright.append((index[1],index[0])) 

        preVal = x
        totalCount+=1

    lineListleft.reverse()
    lineListright.reverse()

    #get length of shortest side 
    length = 0
    if(len(lineListleft) > len(lineListright)):
        length = len(lineListright) - maxVisionDistance
    else:
        length = len(lineListleft) - maxVisionDistance
        
    centerPointsY = []
    centerPointsX = []

    #filter bad points and draw center/side lines
    leftpoint = (0,0)
    rightpoint = (0,0)
    for p in range(length):

        if calculateDistance(lineListleft[p][0], lineListleft[p][1], lineListleft[p+1][0], lineListleft[p+1][1]) < maxDistance:
            leftpoint = lineListleft[p+1]
            cv2.line(img, lineListleft[p], lineListleft[p+1], [0, 255, 0], 3)

        if calculateDistance(lineListright[p][0], lineListright[p][1], lineListright[p+1][0], lineListright[p+1][1]) < maxDistance:
            rightpoint = lineListright[p+1]
            cv2.line(img, lineListright[p], lineListright[p+1], [255, 0, 0], 3)
        
        roadWidth = abs(leftpoint[0] - rightpoint[0])

        if roadWidth > minRoadWidth:
            centerPointsX.append(leftpoint[0] + round(roadWidth / 2))
            centerPointsY.append(leftpoint[1])

    #moving average filter on centerline
    filteredCenterPointsX = []
    if len(centerPointsX) > 10:
        filteredCenterPointsX = movingaverage(centerPointsX, 30)

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

