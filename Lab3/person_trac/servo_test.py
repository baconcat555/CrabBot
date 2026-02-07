import serial
import time
n = 0
nn = str(n) + "\n"
if __name__== '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()
    while True:
        ser.write(nn.encode('utf-8'))
        n += 1
        nn = str(n) + "\n"
        # if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        time.sleep(1)





# time.sleep(2)

# while True:
#     message = f"{5},{6}\n"
#     ser.write(message.encode())

#     time.sleep(1)
#     print(ser.readline())
    