from vehicleSerial import *
import time
from fileWriter import *
from simple_pid import PID

steeringController = PID(0.4, 0.1, 0.05, setpoint=0)
throttleController = PID(0.8, 0.1, 0.05, setpoint=0)
steeringController.sample_time = 0.5  # update every 0.1 seconds
throttleController.sample_time = 0.5  # update every 0.1 seconds
steeringController.output_limits = (-1,1)
throttleController.output_limits = (0,1)


desiredSteeringAngle = 0  # from -1 to 1
desiredThrottle= 0  # from 0 to 1
currentSteeringAngle = 0

steeringAgressivity = 0.9
maxSteeringAgressivity = 0.8

throttleAgressivity = 0.6
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

def getCurrentStatus():
    global currentThrottle, currentSteeringAngle
    return currentThrottle, currentSteeringAngle

#TODO implement PID for steering and throttle

def PID(debug):
    errorMessage = ""
    empty = 0
    global currentThrottle

    steeringController.setpoint = desiredSteeringAngle

    currentSteeringAngle, empty = readValue()
    steeringCommand = steeringController(currentSteeringAngle)

    leftCommand = 0
    rightCommand = 0

    if steeringCommand < 0:
        #go right
        rightCommand = steeringCommand * -1

    elif steeringCommand > 0:
        #go left
        leftCommand = steeringCommand 

    throttleController.setpoint = desiredThrottle

    throttleCommand = throttleController(currentThrottle)

    if debug == "True":
        debugMessage = ("readed throttle = " + str(currentThrottle) + " readed steeringangle = " + str(currentSteeringAngle) + " leftCommand = " + str(leftCommand) + " rightCommand = " + str(rightCommand) + " desiredSteeringAngle = " + str(desiredSteeringAngle) + " throttleCommand = " + str(throttleCommand))
        writeControlMessage(debugMessage)
    
    errorMessage = writeValue(leftCommand, rightCommand, throttleCommand)
    currentThrottle = throttleCommand

def control(debug):
    errorMessage = ""
    empty = 0
    global currentThrottle, currentSteeringAngle

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

    errorMessage = writeValue(leftCommand, rightCommand, throttleCommand)

    if debug == "True":
        debugMessage = ("steeringError = " +str(steeringError) + " readed throttle = " + str(currentThrottle) + " readed steeringangle = " + str(currentSteeringAngle) + " leftCommand = " + str(leftCommand) + " rightCommand = " + str(rightCommand) + " desiredSteeringAngle = " + str(desiredSteeringAngle) + " throttleError = " + str(throttleError) + " throttleCommand = " + str(throttleCommand))
        writeControlMessage(debugMessage)
    time.sleep(0.1)

def main(debug):

    while True:
        control(debug)
        #PID(debug)


