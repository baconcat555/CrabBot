#include <AccelStepper.h>
#include <math.h>

#define STEP_PIN    18
#define DIR_PIN     19
#define INDEX_PIN   4
#define LED_PIN     2

#define ENC_A_PIN   36
#define ENC_B_PIN   39

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// =============================
// Globals
// =============================
volatile long encoderCount = 0;
volatile bool indexDetected = false;

bool homed = false;
unsigned long lastMilli = 0;

// =============================
// Geometry
// =============================
// Stepper commanded resolution
const float STEPS_PER_REV = 400.0f;
const float STEPS_PER_DEG = STEPS_PER_REV / 360.0f;

// Encoder measured resolution
// You measured about 1960 counts/rev, so use 2000 for now
const float ENCODER_COUNTS_PER_REV = 2000.0f;
const float ENCODER_COUNTS_PER_DEG = ENCODER_COUNTS_PER_REV / 360.0f;

// =============================
// Motion settings
// =============================
const float HOME_SPEED_STEPS   = 300.0f;
const float MOVE_SPEED_STEPS   = 500.0f;
const float RETURN_SPEED_STEPS = 500.0f;

// Fixed start position after homing
const float START_POS_DEG = -10.0f;

// Hard upper bound
const float SEARCH_MAX_DEG = 133.5f;

// Pause
const unsigned long DWELL_MS = 50;
const unsigned long CYCLE_PAUSE_MS = 200;

// Feedback tolerance in encoder counts
const long ENC_TOL_COUNTS = 2;

// =============================
// Trajectory (already relative to HOME)
// =============================
const int NPTS = 4;
float pos_wp_deg[NPTS] = {3.91f, 35.58f, 66.17f, 100.78f};
float t_wp[NPTS]       = {0.000f, 0.232f, 0.463f, 0.694f};

long targetEncCounts[NPTS];
float segmentSpeedSteps[NPTS - 1];

// =============================
// Utility
// =============================
long degToSteps(float deg) {
  return lroundf(deg * STEPS_PER_DEG);
}

long degToEncCounts(float deg) {
  return lroundf(deg * ENCODER_COUNTS_PER_DEG);
}

float degPerSecToStepsPerSec(float deg_per_sec) {
  return deg_per_sec * STEPS_PER_DEG;
}

// =============================
// Encoder + Index ISRs
// =============================
// Simple 2x decoding using A channel interrupt
void IRAM_ATTR encoderAISR() {
  int a = digitalRead(ENC_A_PIN);
  int b = digitalRead(ENC_B_PIN);

  if (a == b) {
    encoderCount++;
  } else {
    encoderCount--;
  }
}

void IRAM_ATTR indexISR() {
  indexDetected = true;
}

// =============================
// Automatic homing
// Rotate until index is seen, then encoder/home = 0
// =============================
void autohome() {
  Serial.println("Going home...");

  indexDetected = false;
  stepper.setSpeed(HOME_SPEED_STEPS);

  while (!indexDetected) {
    stepper.runSpeed();
  }

  stepper.setSpeed(0);
  encoderCount = 0;
  stepper.setCurrentPosition(0);
  homed = true;

  Serial.println("Homing complete. Home = 0");
}

// =============================
// Feedback move:
// drive until encoder reaches target
// =============================
void moveToWithFeedback(long targetEnc, float speedStepsPerSec) {
  while (true) {
    long err = targetEnc - encoderCount;

    if (labs(err) <= ENC_TOL_COUNTS) {
      stepper.setSpeed(0);
      break;
    }

    if (err > 0) {
      stepper.setSpeed(fabs(speedStepsPerSec));
    } else {
      stepper.setSpeed(-fabs(speedStepsPerSec));
    }

    stepper.runSpeed();
  }
}

// =============================
// Prepare trajectory
// =============================
bool prepareTrajectory() {
  for (int i = 0; i < NPTS; i++) {
    if (pos_wp_deg[i] < START_POS_DEG || pos_wp_deg[i] > SEARCH_MAX_DEG) {
      Serial.println("Trajectory angle outside allowed range.");
      return false;
    }

    targetEncCounts[i] = degToEncCounts(pos_wp_deg[i]);
  }

  for (int i = 0; i < NPTS - 1; i++) {
    float dt = t_wp[i + 1] - t_wp[i];
    float dtheta = pos_wp_deg[i + 1] - pos_wp_deg[i];

    if (dt <= 0.0f) {
      Serial.println("Invalid waypoint times.");
      return false;
    }

    segmentSpeedSteps[i] = degPerSecToStepsPerSec(dtheta / dt);
  }

  return true;
}

// =============================
// One trajectory cycle
// =============================
void runOneTrajectoryCycle() {
  // Go to fixed start position = -10 deg
  moveToWithFeedback(degToEncCounts(START_POS_DEG), RETURN_SPEED_STEPS);
  delay(DWELL_MS);

  // Go to first waypoint
  moveToWithFeedback(targetEncCounts[0], MOVE_SPEED_STEPS);
  delay(DWELL_MS);

  // Timed segments
  for (int i = 0; i < NPTS - 1; i++) {
    moveToWithFeedback(targetEncCounts[i + 1], segmentSpeedSteps[i]);
    delay(DWELL_MS);
  }

  // Pause at last point
  delay(CYCLE_PAUSE_MS);

  // Return to fixed start position
  moveToWithFeedback(degToEncCounts(START_POS_DEG), RETURN_SPEED_STEPS);
  delay(CYCLE_PAUSE_MS);
}

// =============================
// Setup
// =============================
void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(INDEX_PIN, INPUT_PULLUP);

  // GPIO 36 and 39 are input-only
  pinMode(ENC_A_PIN, INPUT);
  pinMode(ENC_B_PIN, INPUT);

  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), encoderAISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(INDEX_PIN), indexISR, RISING); // change to FALLING if needed

  stepper.setMaxSpeed(2000);

  autohome();

  if (!homed) {
    while (1) {
    }
  }

  if (!prepareTrajectory()) {
    while (1) {
    }
  }

  digitalWrite(LED_PIN, HIGH);
}

// =============================
// Loop
// =============================
void loop() {
  runOneTrajectoryCycle();

  if (millis() - lastMilli > 100) {
    Serial.print("Encoder: ");
    Serial.print(encoderCount);
    Serial.print("  StepperPos: ");
    Serial.println(stepper.currentPosition());
    lastMilli = millis();
  }
}



// volatile long encoderCount = 0;

// #define ENC_A_PIN 36
// #define ENC_B_PIN 39

// void IRAM_ATTR encoderAISR() {
//   int a = digitalRead(ENC_A_PIN);
//   int b = digitalRead(ENC_B_PIN);

//   if (a == b) encoderCount++;
//   else encoderCount--;
// }

// void setup() {
//   Serial.begin(115200);
//   pinMode(ENC_A_PIN, INPUT);
//   pinMode(ENC_B_PIN, INPUT);
//   attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), encoderAISR, CHANGE);
// }

// void loop() {
//   static long last = 999999;
//   if (encoderCount != last) {
//     Serial.println(encoderCount);
//     last = encoderCount;
//   }
// }
