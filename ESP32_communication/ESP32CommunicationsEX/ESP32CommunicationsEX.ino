#include "Wire.h"

#define I2C_DEV_ADDR 0x55

// Example position variable (0–255)
uint8_t position = 0;

// Received values
uint8_t state = 0;
uint8_t rate  = 0;

void onRequest() {
  // Send a single byte (position)
  Wire.write(position);
}

void onReceive(int len) {
  if (len < 2) {
    Serial.println("Error: Not enough data");
    while (Wire.available()) Wire.read();
    return;
  }

  state = Wire.read();
  rate  = Wire.read();

  while (Wire.available()) Wire.read();

  Serial.printf("Received -> State: %d, Rate: %d\n", state, rate);
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  Wire.onReceive(onReceive);
  Wire.onRequest(onRequest);
  Wire.begin((uint8_t)I2C_DEV_ADDR);
}

void loop() {
  // Example: update position (replace with real sensor/encoder logic)
  position++;
  delay(100);
}