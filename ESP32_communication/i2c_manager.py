from smbus2 import SMBus
import time

class ESP32Device:
    def __init__(self, address, name=None,length=4):
        self.address = address
        self.length = length
        self.name = name or f"ESP32_{hex(address)}"


    # This allows the device to be printed
    def __repr__(self):
        return f"<ESP32Device name={self.name} address={hex(self.address)}>"


class I2CManager:
    def __init__(self, bus_id=1):
        self.bus = SMBus(bus_id)
        #dict so can lookup via address
        self.devices = {}

    def add_device(self, address, name=None,length=4):
        if address in self.devices:
            #if the device has already been added
            #trigger exception
            raise ValueError(f"Device at address {hex(address)} already exists")
        device = ESP32Device(address, name,length)
        #add to dict
        self.devices[address] = device
        return device

    def remove_device(self, address):
        #Don't do this one, may be good for debugging, who knows
        if address in self.devices:
            del self.devices[address]

    def list_devices(self):
        return list(self.devices.values())

    def write(self, address, data):
        #Send data to the ESP32 raw bytes, triggers ".onRecieve" Function this event is an INTERUPT

        if address not in self.devices:
            raise ValueError("Device not registered")

        try:
            #0x00 is the command or register bit, we can maybe use, but not necessarily possible or needed
            #Data must be provided in the format data = [255,0,128,42] (each is a byte) due to i2c restrictions.
            #Data needs to be reconstructed on ESP32 side in bytes.

            self.bus.write_i2c_block_data(address, 0x00, data)
        except Exception as e:
            print(f"Write error to {hex(address)}: {e}")

    def read(self, address):
        #will trigger ".onrRequest" function on ESP32 on the ESP32 this event is an INTERUPT
        if address not in self.devices:
            raise ValueError("Device not registered")

        try:
            return self.bus.read_i2c_block_data(address, 0x00, self.devices[address].length)
        except Exception as e:
            print(f"Read error from {hex(address)}: {e}")
            return None

    def write_read(self, address, write_data, read_length, delay=0.01):
        #for sensors or feedback, note the delay
        self.write(address, write_data)
        time.sleep(delay)
        return self.read(address, read_length)

    def ping(self, address):
        #Check address
        try:
            self.bus.write_quick(address)
            return True
        except:
            return False

    def scan(self):
        #version of the scan from smbus2 in command line

        found = []
        for addr in range(0x03, 0x77):
            try:
                self.bus.write_quick(addr)
                found.append(addr)
            except:
                pass
        return found

    def close(self):
        #if it isn't closed it may have problems when restarting
            self.bus.close()