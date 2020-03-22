from vehicleSerial import *
import time

minSteerSensorValue = 500
maxSteerSensorValue = 0

minThrottleSensorValue = 500
maxThrottleSensorValue = 0

#connect to autonomous vehicle
connectionResult = connect('/dev/ttyUSB0')

while connectionResult != "succes":
    print("arduino failed to connect but trying again...")
    connectionResult = connect('/dev/ttyUSB0')

print("arduino connected!")

time.sleep(3)

print("please turn the steeringwheel and throttle multiple times from minimum to maximum states. Then close this program")

time.sleep(4)

while True:
    currentSteeringAngle, currentThrottle = readValue()

    if currentSteeringAngle != None or currentThrottle != None:

        if currentSteeringAngle < minSteerSensorValue:
            minSteerSensorValue = currentSteeringAngle
        elif currentSteeringAngle > maxSteerSensorValue:
            maxSteerSensorValue = currentSteeringAngle
        
        if currentThrottle < minThrottleSensorValue:
            minThrottleSensorValue = currentThrottle
        elif currentThrottle > maxThrottleSensorValue:
            maxThrottleSensorValue = currentThrottle

        print("minSteerSensorValue > " + str(minSteerSensorValue) + " maxSteerSensorValue > " + str(maxSteerSensorValue) + " minThrottleSensorValue > " + str(minThrottleSensorValue) + " maxThrottleSensorValue > " + str(maxThrottleSensorValue))

    time.sleep(0.1) 