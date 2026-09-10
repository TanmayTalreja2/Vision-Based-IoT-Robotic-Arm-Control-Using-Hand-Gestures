import serial
import time

# CHANGE COM3 IF YOUR ARDUINO USES A DIFFERENT PORT
arduino = serial.Serial(
    "COM4",
    9600,
    timeout=1
)

# Arduino resets when serial connection opens
time.sleep(2)

print("Connected to Arduino")

while True:

    command = input(
        "Enter Base,Elbow,Gripper "
        "(example: 90,120,10): "
    )

    if command.lower() == "q":
        break

    arduino.write(
        (command + "\n").encode()
    )

    time.sleep(0.1)

    if arduino.in_waiting > 0:
        response = arduino.readline().decode().strip()
        print("Arduino:", response)


arduino.close()

print("Disconnected")