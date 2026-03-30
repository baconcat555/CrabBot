from i2c_manager import I2CManager

def main():
    # Initialize manager
    #set bus_id pis use 1.
    Messenger = I2CManager(bus_id=1)

    try:
        # Scan for devices
        found_devices = Messenger.scan()
        print("I2C devices found:", [hex(addr) for addr in found_devices])

        #add devices, in the actual system there will be 3, and a turret.
        Messenger.add_device(found_devices[0],"motor_controller1",4)
        Messenger.add_device(found_devices[1],"motor_controller2",4)

        # List devices
        print("Devices:")
        for dev in Messenger.list_devices():
            print(dev)



        # Ping
        print("\nPing results:")
        for addr in Messenger.devices:
            status = Messenger.ping(addr)
            print(f"{hex(addr)}:", "OK" if status else "No response")

        Messenger.read(found_devices[0])
    finally:
        # close the bus or it breaks
        Messenger.close()


# apparently this is a good best practice idk?
if __name__ == "__main__":
    main()