// DM556T (STEP/DIR only) with Arduino UNO/ELEGOO
// Wiring (common-cathode):
// PUL+ -> D8
// PUL- -> Arduino GND
// DIR+ -> D6
// DIR- -> Arduino GND
// ENA not connected







const int STEP_PIN = 6;   // PUL+
const int DIR_PIN  = 8;   // DIR+

// "Speed" is step pulse frequency (NOT throttle). Start conservative.
const unsigned long STEP_FREQ_HZ = 2000;     // steps/sec
const unsigned long RUN_TIME_MS  = 60000;     // 5 seconds

// Datasheet: pulse width >= 2.5us, use 5us safely
const unsigned int PULSE_HIGH_US = 10;

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, HIGH);   // direction (change to LOW to reverse)

  delay(10); // allow DIR to settle before stepping
}

void loop() {
  const unsigned long period_us = 1000000UL / STEP_FREQ_HZ;
  const unsigned long low_us    = (period_us > PULSE_HIGH_US) ? (period_us - PULSE_HIGH_US) : 1;

  while (true) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(PULSE_HIGH_US);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds((unsigned int)low_us);
  }
}
