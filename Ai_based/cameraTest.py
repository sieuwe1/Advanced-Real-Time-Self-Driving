import numpy as np
import cv2

#cap = cv2.VideoCapture('http://root:root@192.168.70.52/mjpg/1/video.mjpg')
cap = cv2.VideoCapture(0)
 
def adjust_gamma(image, gamma):
    image = cv2.resize(image,(512,256))
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
    
while(True):

    ret, frame = cap.read()
    cv2.imshow('original',frame)

    cv2.imshow('Gamma',adjust_gamma(frame, 2))

    frame= cv2.add(frame,np.array([20.0]))
    cv2.imshow('brighter',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()