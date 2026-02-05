from gpiozero import AngularServo
from time import sleep

# The default GPIO Zero servo range works well with 9g servos
# Connect Signal to GPIO 17 (Pin 11)
#servo_x = AngularServo(23, min_pulse_width=0.0006, max_pulse_width=0.0024)
servo_y = AngularServo(24, min_pulse_width=0.0006, max_pulse_width=0.0024)

try:
    while True:
        print("Set to Minimum")
        servo_y.angle = 45 # Moves to 0 degrees
        sleep(1)
        
        #print("Set to Maximum")
        #myServo.max() # Moves to 180 degrees
        #sleep(1)
except KeyboardInterrupt:
    print("Program stopped")
