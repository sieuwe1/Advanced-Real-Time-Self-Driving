from vehicleSerial import *
import time

desiredSteeringAngle = 0  # from -1 to 1
desiredThrottle= 0.3  # from 0 to 1

steeringAgressivity = 0.9
maxSteeringAgressivity = 0.8

throttleAgressivity = 0.5
maxthrottleAgressivity = 0.8
currentThrottle = 0

def setDesiredSteeringAngle(steeringAngle):
    global desiredSteeringAngle
    desiredSteeringAngle = steeringAngle
def getDesiredSteeringAngle():
    global desiredSteeringAngle
    return desiredSteeringAngle

def setDesiredThrottle(throttle):
    global desiredThrottle
    desiredThrottle = throttle

def getDesiredThrottle():
    global desiredThrottle
    return desiredThrottle


#TODO implement PID for steering and throttle

def main():
    errorMessage = ""
    empty = 0
    global currentThrottle

    while errorMessage != "error":
        currentSteeringAngle, empty = readValue()
        
        steeringError = desiredSteeringAngle - currentSteeringAngle
        throttleError = desiredThrottle - currentThrottle

        steeringCommand = 0
        throttleCommand = 0    

        #steeringCommand = currentSteeringAngle + (steeringError * steeringAgressivity)
        steeringCommand = steeringError * steeringAgressivity
        if steeringCommand > maxSteeringAgressivity:
            steeringCommand = maxSteeringAgressivity
        elif steeringCommand < (maxSteeringAgressivity * -1):
            steeringCommand = maxSteeringAgressivity * -1

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

        throttleCommand = throttleError * throttleAgressivity
        if throttleCommand > maxSteeringAgressivity:
            throttleCommand = maxSteeringAgressivity

        if throttleCommand < 0:
            throttleCommand = 0

        currentThrottle = throttleCommand

        print("steeringError = " +str(steeringError) + " leftCommand = " + str(leftCommand) + " rightCommand = " + str(rightCommand) + " desiredSteeringAngle = " + str(desiredSteeringAngle) + " throttleError = " + str(throttleError) + " throttleCommand = " + str(throttleCommand))
        errorMessage = writeValue(leftCommand, rightCommand, throttleCommand)
        
        time.sleep(0.1)
    


