from vehicleSerial import *
from control import *
import time
import threading

#connect to autonomous vehicle
connectionResult = connect('/dev/ttyUSB0')

while connectionResult != "succes":
    print("arduino failed to connect but trying again...")
    connectionResult = connect('/dev/ttyUSB0')

print("arduino connected!")

time.sleep(3)

debugString = False

controlThread = threading.Thread(target=main, args=(debugString,))
controlThread.start()
print("starting...")

while(True):
    setDesiredSteeringAngle(-0.8)
    setDesiredThrottle(1)
    time.sleep(2.5)
    print("switch")
    setDesiredSteeringAngle(0.8)
    setDesiredThrottle(0)
    time.sleep(2.5)
    print("switch")
