#include <AccelStepper.h>

// =============================
// Pins
// =============================
#define STEP_PIN   18
#define DIR_PIN    19

#define ENC_A_PIN  36
#define ENC_B_PIN  32

#define LED_PIN    2

// =============================
// Direction control
// =============================
#define DIR_SIGN 1   // 

// =============================
// Stepper
// =============================
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// =============================
// Encoder
// =============================
volatile long encoderCount = 0;

// =============================
// Calibration
// =============================
long encMin = 0;
long encMax = 0;

bool maxCaptured = false;
bool readyToRun = false;

// =============================
// Geometry
// =============================
const float MAX_PHYSICAL_DEG = 160.5f;

// =============================
// Motion settings
// =============================
const float MOVE_SPEED_STEPS   = 500.0f;
const float RETURN_SPEED_STEPS = 200.0f;

const long ENC_TOL_COUNTS = 2;

// =============================
// Trajectory
// =============================
const int NPTS = ;
float pos_wp_deg[NPTS] = {3.91f, 35.58f, 66.17f, 100.78f};
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

  if (a == b) encoderCount++;
  else encoderCount--;
}

// =============================
// Utility
// =============================
float countsPerDegree() {
  return (float)encMax / MAX_PHYSICAL_DEG;
}

long degToEncCounts(float deg) {
  return lroundf(deg * countsPerDegree());
}

float degPerSecToStepsPerSec(float deg_per_sec) {
  const float STEPS_PER_REV = 400.0f;
  const float STEPS_PER_DEG = STEPS_PER_REV / 360.0f;
  return deg_per_sec * STEPS_PER_DEG;
}

// =============================
// Feedback move (DIR_SIGN applied)
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

    segmentSpeedSteps[i] = degPerSecToStepsPerSec(dtheta / dt);
  }

  return true;
}

// =============================
// One trajectory cycle
// =============================
void runOneTrajectoryCycle() {
  moveToWithFeedback(0, RETURN_SPEED_STEPS);   // go to min
  delay(50);

  moveToWithFeedback(targetEncCounts[0], MOVE_SPEED_STEPS);
  delay(50);

  for (int i = 0; i < NPTS - 1; i++) {
    moveToWithFeedback(targetEncCounts[i + 1], segmentSpeedSteps[i]);
    delay(50);
  }

  delay(200);

  moveToWithFeedback(0, RETURN_SPEED_STEPS);   // return to min
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

  stepper.setMaxSpeed(2000);

  encoderCount = 0;
  encMin = 0;

  Serial.println("=== CALIBRATION ===");
  Serial.println("Move to MAX and press 'm'");
}

// =============================
// Loop
// =============================
void loop() {
  static unsigned long lastPrint = 0;

  // Print encoder count every 100 ms
  if (millis() - lastPrint > 100) {
    Serial.print("Encoder: ");
    Serial.println(encoderCount);
    lastPrint = millis();
  }
  if (state == CALIBRATING_MAX) {
    if (Serial.available()) {
      char c = Serial.read();

      if (c == 'm') {
        encMax = encoderCount;
        maxCaptured = true;

        Serial.print("MAX captured: ");
        Serial.println(encMax);

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
          while (1);
        }

        digitalWrite(LED_PIN, HIGH);
        state = RUNNING;

        Serial.println("Running trajectory...");
      }
    }
  }

  else if (state == RUNNING) {
    runOneTrajectoryCycle();
  }
}
