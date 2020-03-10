import sys
from pyfirmata import Arduino, util
import time

mapValues = True 

steerSensorPin = 0
minSteerSensorValue = 0.7644
maxSteerSensorValue = 0.8925

throttleSensorPin = 1
minThrottleSensorValue = 0.173
maxThrottleSensorValue = 0.752

minThrottlePWM = 0 
maxThrottlePWM = 200 #real max 256

minSteeringPWM = -200 #real min 256
maxSteeringPWM = 200 #real max 256

arduinoNano = None
leftPin = None
rightPin = None 
throttlePin = None

def connect(port):
    global arduinoNano, leftPin, rightPin, throttlePin
    if arduinoNano == None:
        arduinoNano = Arduino(port)
        it = util.Iterator(arduinoNano)
        it.start()
        arduinoNano.analog[steerSensorPin].enable_reporting()
        arduinoNano.analog[throttleSensorPin].enable_reporting()
        throttlePin = arduinoNano.get_pin('d:6:p')
        leftPin = arduinoNano.get_pin('d:10:p')
        rightPin = arduinoNano.get_pin('d:11:p')
        return "succes"
    else:
        return "error"

def disconnect():
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
                mappedSteeringPosition = map(steerPosition, minSteerSensorValue, maxSteerSensorValue, -1, 1)
                mappedThrottlePosition = map(throttlePosition, minThrottleSensorValue, maxThrottleSensorValue, 0, 1)
                print("readed steeringposition > " + str(mappedSteeringPosition) + " readed throttle " + str(mappedThrottlePosition))
                return((mappedSteeringPosition,mappedThrottlePosition))
            else:
                print("readed steeringposition > " + str(steerPosition) + " readed throttle " + str(throttlePosition))
                return((steerPosition,throttlePosition))
    else:
        return("error")

def writeValue(left,right,throttle):
    global leftPin, rightPin, throttlePin
    mappedLeft = map(left, -1, 1, minSteeringPWM, maxSteeringPWM)
    mappedRight = map(right, -1, 1, minSteeringPWM, maxSteeringPWM)
    mappedThrottle = map(throttle, 0, 1, minThrottlePWM, maxThrottlePWM)
    print("writing values left> " + str(mappedLeft) + " right> " + str(mappedRight) + " throttle> " + str(mappedThrottle))
    leftPin.write(mappedLeft)
    rightPin.write(mappedRight)
    throttlePin.write(mappedThrottle)
    

def map (value, fromSource,  toSource,  fromTarget,  toTarget):
    return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget


#connect('/dev/ttyUSB0')
#while(True):
    #writeValue(1,0)
    #time.sleep(10)
    #disconnect()
