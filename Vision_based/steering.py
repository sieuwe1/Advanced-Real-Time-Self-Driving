import numpy as np
import cv2
import math
from collections import deque

prevDeltas = deque(maxlen = 6)

def getSteeringCommand(windowCenter, centerPoints, img):
    
    shortestDelta = 999
    
    for centerPoint in centerPoints:
        delta = round(calculateDistance(windowCenter[0], windowCenter[1], centerPoint[0], centerPoint[1]),2)
        if shortestDelta > delta:
            shortestDelta = delta
    
    
    a = (centerPoints[-1][0],centerPoints[-1][1])
    b = (centerPoints[0][0],centerPoints[0][1])

    shortestDelta *= np.sign((b[0] - a[0]) * (windowCenter[1] - a[1]) - (b[1] - a[1]) * (windowCenter[0] - a[0])) 

    prevDeltas.append(shortestDelta)
    
    avgShortestDelta = round (sum(prevDeltas) / len(prevDeltas))

    cv2.putText(img,"Delta: " + str(avgShortestDelta),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),4,cv2.LINE_AA)

    cv2.circle(img,windowCenter,5,(0,255,255),-1)
       
    return img

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist
