// DM556T (STEP/DIR only) with Arduino UNO/ELEGOO
// Wiring (common-cathode):
// PUL+ -> D8
// PUL- -> Arduino GND
// DIR+ -> D6
// DIR- -> Arduino GND
// ENA not connected
#include <QuadratureEncoder.h>
// must also have enableInterrupt.h library

// Use any 2 pins for interrupt, this utilizes EnableInterrupt Library. 
// Even analog pins can be used. A0 = 14,A1=15,..etc for arduino nano/uno

// Max number of Encoders object you can create is 4. This example only uses 2.

Encoders leftEncoder(2,3);	// Create an Encoder object name leftEncoder, using digitalpin 2 & 3
 // Encoder object name rightEncoder using analog pin A0 and A1 
const int STEP_PIN = 6;   // PUL+
const int DIR_PIN  = 8;   // DIR+

// "Speed" is step pulse frequency (NOT throttle). Start conservative.
const unsigned long STEP_FREQ_HZ = 1800;     // steps/sec
const unsigned long RUN_TIME_MS  = 60000;     // 5 seconds

// Datasheet: pulse width >= 2.5us, use 5us safely
const unsigned int PULSE_HIGH_US = 10;

void setup() {
  Serial.begin(9600);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, HIGH);   // direction (change to LOW to reverse)

  delay(10); // allow 
}


unsigned long lastMilli = 0;

void loop() {
  const unsigned long period_us = 1000000UL / STEP_FREQ_HZ;
  const unsigned long low_us    = (period_us > PULSE_HIGH_US) ? (period_us - PULSE_HIGH_US) : 1;

  while (true) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(PULSE_HIGH_US);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds((unsigned int)low_us);
  }
  // put your main code here, to run repeatedly:
  // print encoder count every 50 millisecond
  if(millis()-lastMilli > 50){ 
    
    long currentLeftEncoderCount = leftEncoder.getEncoderCount();
    // long currentRightEncoderCount = rightEncoder.getEncoderCount();
    
    Serial.print(currentLeftEncoderCount);
    Serial.print(" , ");
    //Serial.println(currentRightEncoderCount);
    
    lastMilli = millis();
  }
   
}



