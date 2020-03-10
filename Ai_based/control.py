from vehicleSerial import *
import time

desiredSteeringAngle = 0  # from -1 to 1
desiredThrottle= 0  # from 0 to 1

steeringAgressivity = 0.1
maxSteeringAgressivity = 0.2

throttleAgressivity = 0.1
maxthrottleAgressivity = 0.2

def setDesiredSteeringAngle(steeringAngle):
    desiredSteeringAngle = steeringAngle
def getDesiredSteeringAngle():
    return desiredSteeringAngle

def setDesiredThrottle(throttle):
    desiredThrottle = throttle

def getDesiredThrottle():
    return desiredThrottle


#TODO implement PID for steering and throttle

def main():
    errorMessage = ""
    while errorMessage != "error":
        currentSteeringAngle, currentThrottle = readValue()
        
        steeringError = desiredSteeringAngle - currentSteeringAngle
        throttleError = desiredThrottle - currentThrottle

        steeringCommand = 0
        throttleCommand = 0    

        steeringCommand = currentSteeringAngle + (steeringError * steeringAgressivity)
        if steeringCommand > maxthrottleAgressivity:
            steeringCommand = maxthrottleAgressivity
        elif steeringCommand < (maxthrottleAgressivity * -1):
            steeringCommand = maxthrottleAgressivity * -1

        leftCommand = 0
        rightCommand = 0

        if steeringCommand < 0:
            #go right
            rightCommand = steeringCommand * -1

        elif steeringCommand > 0:
            #go left
            leftCommand = steeringCommand 

        else:
            print("no steering needed")

        throttleCommand = currentThrottle + (throttleError * throttleAgressivity)
        if throttleCommand > maxSteeringAgressivity:
            throttleCommand = maxSteeringAgressivity

        if throttleCommand < 0:
            throttleCommand = 0

        print("leftCommand = " + str(leftCommand) + " rightCommand = " + str(rightCommand))
        errorMessage = writeValue(leftCommand, rightCommand, throttleCommand)
        
        time.sleep(0.1)
    


