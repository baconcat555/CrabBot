#include <AccelStepper.h>

// =============================
// Pins
// =============================
#define STEP_PIN   12
#define DIR_PIN    14

#define ENC_A_PIN  35
#define ENC_B_PIN  34

#define LED_PIN    2

// =============================
// Direction control
// Change to -1 if motor goes opposite
// =============================
#define DIR_SIGN -1

// =============================
// Stepper
// =============================
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// =============================
// Encoder
// =============================
volatile long encoderCount = 0;

// =============================
// Calibration / state
// =============================
long encMin = 0;   // always treated as 0
long encMax = 0;

bool maxCaptured = false;
bool readyToRun = false;

// =============================
// Geometry
// Your measured relation:
// angle_deg = (encoderCount * 12) / 1960
// Therefore:
// counts_per_deg = 1960 / 12
// =============================
const float COUNTS_PER_12_DEG = 1960.0f;
const float DEGREES_PER_1960_COUNTS = 12.0f;
const float COUNTS_PER_DEG = COUNTS_PER_12_DEG / DEGREES_PER_1960_COUNTS;

// Hard maximum allowed leg angle
const float MAX_PHYSICAL_DEG = 60.0f;

// =============================
// Motor / gearbox settings
// =============================
const float MOTOR_STEPS_PER_REV = 400.0f;
const float GEAR_RATIO = 30.0f;

// =============================
// Motion settings
// =============================
const float MAX_STEP_SPEED    = 1300.0f;
const float MOVE_SPEED_STEPS  = 1300.0f;
const float RETURN_SPEED_STEPS = 1300.0f;

const long ENC_TOL_COUNTS = 10;

// =============================
// Trajectory
// Angles are leg angles from MIN position
// =============================
const int NPTS = 4;
float pos_wp_deg[NPTS] = {3.55f, 15.45f, 35.54f, 53.89f};
float t_wp[NPTS]       = {0.000f, 0.232f, 0.463f, 0.694f};

long targetEncCounts[NPTS];
float segmentSpeedSteps[NPTS - 1];

// =============================
// State machine
// =============================
enum State {
  CALIBRATING_MAX,
  WAITING_FOR_START,
  RUNNING
};

State state = CALIBRATING_MAX;

// =============================
// Encoder ISR
// =============================
void IRAM_ATTR encoderISR() {
  int a = digitalRead(ENC_A_PIN);
  int b = digitalRead(ENC_B_PIN);

  // Flip these two lines if encoder direction is reversed
  if (a == b) encoderCount--;
  else encoderCount++;
}

// =============================
// Utility
// =============================

// Convert encoder counts -> leg angle
float encCountsToDeg(long counts) {
  return (counts * DEGREES_PER_1960_COUNTS) / COUNTS_PER_12_DEG;
}

// Convert leg angle -> encoder counts
long degToEncCounts(float deg) {
  return lroundf(deg * COUNTS_PER_DEG);
}

// Convert leg angular speed -> motor step rate
// IMPORTANT: includes gearbox ratio
float degPerSecToStepsPerSec(float deg_per_sec) {
  const float MOTOR_STEPS_PER_DEG = MOTOR_STEPS_PER_REV / 360.0f;
  return deg_per_sec * GEAR_RATIO * MOTOR_STEPS_PER_DEG;
}

// =============================
// Feedback move (DIR_SIGN applied)
// Encoder count is treated as truth
// =============================
void moveToWithFeedback(long targetEnc, float speedStepsPerSec) {
  while (true) {
    long err = targetEnc - encoderCount;

    if (labs(err) <= ENC_TOL_COUNTS) {
      stepper.setSpeed(0);
      break;
    }

    if (err > 0) {
      stepper.setSpeed(DIR_SIGN * fabs(speedStepsPerSec));
    } else {
      stepper.setSpeed(-DIR_SIGN * fabs(speedStepsPerSec));
    }

    stepper.runSpeed();
  }
}

// =============================
// Prepare trajectory
// =============================
bool prepareTrajectory() {
  for (int i = 0; i < NPTS; i++) {
    if (pos_wp_deg[i] < 0.0f || pos_wp_deg[i] > MAX_PHYSICAL_DEG) {
      Serial.println("Trajectory angle outside range");
      return false;
    }

    targetEncCounts[i] = degToEncCounts(pos_wp_deg[i]);
  }

  for (int i = 0; i < NPTS - 1; i++) {
    float dt = t_wp[i + 1] - t_wp[i];
    float dtheta = pos_wp_deg[i + 1] - pos_wp_deg[i];

    if (dt <= 0.0f) {
      Serial.println("Invalid waypoint timing");
      return false;
    }

    float calculatedSpeed = degPerSecToStepsPerSec(dtheta / dt);

    // Limit segment speed so it never exceeds MAX_STEP_SPEED
    if (fabs(calculatedSpeed) > MAX_STEP_SPEED) {
      segmentSpeedSteps[i] = (calculatedSpeed > 0) ? MAX_STEP_SPEED : -MAX_STEP_SPEED;
    } else {
      segmentSpeedSteps[i] = calculatedSpeed;
    }
  }

  return true;
}

// =============================
// One trajectory cycle
// encoderCount = 0 is treated as MIN
// =============================
void runOneTrajectoryCycle() {
  // Go to min
  moveToWithFeedback(0, RETURN_SPEED_STEPS);
  delay(50);

  // Go to first waypoint
  moveToWithFeedback(targetEncCounts[0], MOVE_SPEED_STEPS);
  delay(50);

  // Timed segments
  for (int i = 0; i < NPTS - 1; i++) {
    moveToWithFeedback(targetEncCounts[i + 1], segmentSpeedSteps[i]);
    delay(50);
  }

  delay(200);

  // Return to min
  moveToWithFeedback(0, RETURN_SPEED_STEPS);
  delay(200);
}

// =============================
// Setup
// =============================
void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(ENC_A_PIN, INPUT);
  pinMode(ENC_B_PIN, INPUT);

  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), encoderISR, CHANGE);

  stepper.setMaxSpeed(5000);

  // Treat current position as MIN
  encoderCount = 0;
  encMin = 0;

  Serial.println("=== CALIBRATION ===");
  Serial.println("Current position is treated as MIN (0 deg).");
  Serial.println("Move to MAX and press 'm'");
}

// =============================
// Loop
// =============================
void loop() {
  static unsigned long lastPrint = 0;

  // Always print encoder count and leg angle
  if (millis() - lastPrint > 100) {
    Serial.print("Encoder: ");
    Serial.print(encoderCount);
    Serial.print("   Angle(deg): ");
    Serial.println(encCountsToDeg(encoderCount));
    lastPrint = millis();
  }

  if (state == CALIBRATING_MAX) {
    if (Serial.available()) {
      char c = Serial.read();

      if (c == 'm') {
        encMax = encoderCount;
        maxCaptured = true;

        Serial.print("MAX captured (counts): ");
        Serial.println(encMax);
        Serial.print("MAX captured (deg): ");
        Serial.println(encCountsToDeg(encMax));

        Serial.println("Move back to MIN and press 's'");
        state = WAITING_FOR_START;
      }
    }
  }

  else if (state == WAITING_FOR_START) {
    if (Serial.available()) {
      char c = Serial.read();

      if (c == 's') {
        encoderCount = 0;

        if (!prepareTrajectory()) {
          while (1) {
          }
        }

        digitalWrite(LED_PIN, HIGH);
        readyToRun = true;
        state = RUNNING;

        Serial.println("Running trajectory...");
      }
    }
  }

  else if (state == RUNNING) {
    runOneTrajectoryCycle();
  }
}
