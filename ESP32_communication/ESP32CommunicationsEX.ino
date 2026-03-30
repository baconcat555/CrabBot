/*********
  Rui Santos & Sara Santos - Random Nerd Tutorials
  Complete project details at https://RandomNerdTutorials.com/esp32-i2c-master-slave-arduino/
  ESP32 I2C Slave example: https://github.com/espressif/arduino-esp32/blob/master/libraries/Wire/examples/WireSlave/WireSlave.ino
*********/

#include "Wire.h"

#define I2C_DEV_ADDR 0x55

uint32_t i = 0;

void onRequest() {
  uint16_t value = i++;
  
  uint8_t data[2];
  data[0] = (value >> 8) & 0xFF;
  data[1] = value & 0xFF;

  Wire.write(data, 2);
}

void onReceive(int len) {
  Serial.printf("Received %d bytes: ", len);

  while (Wire.available()) {
    uint8_t b = Wire.read();
    Serial.printf("0x%02X ", b);
  }

  Serial.println();
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Wire.onReceive(onReceive);
  Wire.onRequest(onRequest);
  Wire.begin((uint8_t)I2C_DEV_ADDR);

/*#if CONFIG_IDF_TARGET_ESP32
  char message[64];
  snprintf(message, 64, "%lu Packets.", i++);
  Wire.slaveWrite((uint8_t *)message, strlen(message));
  Serial.print('Printing config %lu', i);
#endif*/
}

void loop() {
  
}
