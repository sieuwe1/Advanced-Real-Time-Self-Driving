import sys
from pyfirmata import Arduino, util
import time

mapValues = True 

steerSensorPin = 2
#minSteerSensorValue = 0.1253
#maxSteerSensorValue = 0.3795
minSteerSensorValue = 0.119
maxSteerSensorValue = 0.4378

throttleSensorPin = 1
minThrottleSensorValue = 0.1700
maxThrottleSensorValue = 0.7720

minThrottlePWM = 0 
maxThrottlePWM = 1.5 #real max 1

minSteeringPWM = 0 
maxSteeringPWM = 0.6 #real max 1

reverseSteering = -1 # make 1 to reverse steering

arduinoNano = None
leftPWM = None
rightPWM = None 
rightEnable = 3
leftEnable = 5 
throttlePin = None

def connect(port):
    global arduinoNano, leftPWM, rightPWM, leftEnable, rightEnable, throttlePin
    if arduinoNano == None:
        arduinoNano = Arduino(port)
        it = util.Iterator(arduinoNano)
        it.start()
        arduinoNano.analog[steerSensorPin].enable_reporting()
        arduinoNano.analog[throttleSensorPin].enable_reporting()
        leftPWM = arduinoNano.get_pin('d:6:p')
        rightPWM = arduinoNano.get_pin('d:9:p')
        throttlePin = arduinoNano.get_pin('d:10:p')
        arduinoNano.digital[rightEnable].write(1)
        arduinoNano.digital[leftEnable].write(1)
        return "succes"
    else:
        return "error"

def disconnect():
    global arduinoNano
    board.digital[rightEnable].write(0)
    board.digital[leftEnable].write(0)
    writeValue(0,0)
    arduinoNano.exit()


def readValue():
    global arduinoNano
    if arduinoNano != None:
        steerPosition = arduinoNano.analog[steerSensorPin].read()  
        throttlePosition = arduinoNano.analog[throttleSensorPin].read() 
        #TODO translate value from min max to -1 1 and 0 1 for throttle
        if(steerPosition != None and throttlePosition != None):
            if(mapValues):
                mappedSteeringPosition = mapSteerPisition(steerPosition, minSteerSensorValue, maxSteerSensorValue, -1, 1)
                mappedThrottlePosition = mapThrottle(throttlePosition, minThrottleSensorValue, maxThrottleSensorValue, 0, 1)
                #print("readed steeringposition > " + str(mappedSteeringPosition) + " readed throttle " + str(mappedThrottlePosition))
                return((mappedSteeringPosition * reverseSteering,mappedThrottlePosition))
            else:
                #print("readed steeringposition > " + str(steerPosition) + " readed throttle " + str(throttlePosition))
                return((steerPosition,throttlePosition))
    else:
        return("error")

def writeValue(left,right,throttle):
    global leftPWM, rightPWM, leftEnable, rightEnable, throttlePin

    mappedLeft = 0
    mappedRight = 0

    if left > 0:
        rightPWM.write(0)
        mappedLeft = round(map(left, -1, 1, minSteeringPWM, maxSteeringPWM),3)
        leftPWM.write(mappedLeft)
    elif right > 0:
        leftPWM.write(0)
        mappedRight = round(map(right, -1, 1, minSteeringPWM, maxSteeringPWM),3)
        rightPWM.write(mappedRight)
    else:
        print("left right commando incorreect")

    mappedThrottle = round(map(throttle, 0, 1, minThrottlePWM, maxThrottlePWM),3)
    #print("writing values left> " + str(mappedLeft) + " right> " + str(mappedRight) + " throttle> " + str(mappedThrottle))
    throttlePin.write(mappedThrottle)
    

def mapThrottle (value, fromSource,  toSource,  fromTarget,  toTarget):
    mappedThottle = (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget
    if mappedThottle > 1:
        mappedThottle = 1
    elif mappedThottle < 0:
        mappedThottle = 0
    return mappedThottle

def mapSteerPisition (value, fromSource,  toSource,  fromTarget,  toTarget):
    mappedSteerPisition = (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget
    if mappedSteerPisition > 1:
        mappedSteerPisition = 1
    elif mappedSteerPisition < -1:
        mappedSteerPisition = -1
    return mappedSteerPisition

def map(value, fromSource,  toSource,  fromTarget,  toTarget):
    return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget
    

#connect('/dev/ttyUSB0')
#while(True):
    #writeValue(1,0)
    #time.sleep(10)
    #disconnect()
