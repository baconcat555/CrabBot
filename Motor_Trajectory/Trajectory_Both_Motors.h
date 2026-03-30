#include <AccelStepper.h>

// =====================================================
// LEG MOTOR PINS
// =====================================================
#define LEG_STEP_PIN   18
#define LEG_DIR_PIN    19
#define LEG_ENC_A_PIN  36
#define LEG_ENC_B_PIN  32

// =====================================================
// HIP MOTOR PINS
// =====================================================
#define HIP_STEP_PIN   12
#define HIP_DIR_PIN    14
#define HIP_ENC_A_PIN  35
#define HIP_ENC_B_PIN  34

#define LED_PIN        2

// =====================================================
// Direction control
// =====================================================
#define LEG_DIR_SIGN   1
#define HIP_DIR_SIGN  -1

// =====================================================
// Steppers
// =====================================================
AccelStepper legStepper(AccelStepper::DRIVER, LEG_STEP_PIN, LEG_DIR_PIN);
AccelStepper hipStepper(AccelStepper::DRIVER, HIP_STEP_PIN, HIP_DIR_PIN);

// =====================================================
// Encoder counts
// =====================================================
volatile long legEncoderCount = 0;
volatile long hipEncoderCount = 0;

// =====================================================
// Calibration / state
// Both motors start physically at MIN = 0
// =====================================================
long legEncMin = 0;
long legEncMax = 0;
bool legMaxCaptured = false;

long hipEncMin = 0;
long hipEncMax = 0;
bool hipMaxCaptured = false;

bool readyToRun = false;

// =====================================================
// LEG GEOMETRY
// =====================================================
const float LEG_MAX_PHYSICAL_DEG = 133.5f;

// =====================================================
// LEG motion settings
// =====================================================
const float LEG_MOVE_SPEED_STEPS   = 500.0f;
const float LEG_RETURN_SPEED_STEPS = 150.0f;
const long  LEG_ENC_TOL_COUNTS     = 2;

// =====================================================
// LEG trajectory
// =====================================================
const int LEG_NPTS = 4;
float leg_pos_wp_deg[LEG_NPTS] = {3.91f, 35.58f, 66.17f, 100.78f};
float leg_t_wp[LEG_NPTS]       = {0.000f, 0.232f, 0.463f, 0.694f};

long  legTargetEncCounts[LEG_NPTS];
float legSegmentSpeedSteps[LEG_NPTS - 1];

// =====================================================
// HIP GEOMETRY
// angle_deg = (encoderCount * 12) / 1960
// =====================================================
const float HIP_COUNTS_PER_12_DEG = 1960.0f;
const float HIP_DEGREES_PER_1960_COUNTS = 12.0f;
const float HIP_COUNTS_PER_DEG = HIP_COUNTS_PER_12_DEG / HIP_DEGREES_PER_1960_COUNTS;
const float HIP_MAX_PHYSICAL_DEG = 60.0f;

// =====================================================
// HIP motor / gearbox settings
// =====================================================
const float HIP_MOTOR_STEPS_PER_REV = 400.0f;
const float HIP_GEAR_RATIO = 30.0f;

// =====================================================
// HIP motion settings
// =====================================================
const float HIP_MAX_STEP_SPEED      = 1300.0f;
const float HIP_MOVE_SPEED_STEPS    = 1300.0f;
const float HIP_RETURN_SPEED_STEPS  = 1300.0f;
const long  HIP_ENC_TOL_COUNTS      = 10;

// =====================================================
// HIP trajectory
// =====================================================
const int HIP_NPTS = 4;
float hip_pos_wp_deg[HIP_NPTS] = {3.55f, 15.45f, 35.54f, 53.89f};
float hip_t_wp[HIP_NPTS]       = {0.000f, 0.232f, 0.463f, 0.694f};

long  hipTargetEncCounts[HIP_NPTS];
float hipSegmentSpeedSteps[HIP_NPTS - 1];

// =====================================================
// State machine
// =====================================================
enum State {
  WAIT_FOR_LEG_MAX,
  WAIT_FOR_HIP_MAX,
  WAIT_FOR_START,
  RUNNING
};

State state = WAIT_FOR_LEG_MAX;

// =====================================================
// Encoder ISRs
// =====================================================
void IRAM_ATTR legEncoderISR() {
  int a = digitalRead(LEG_ENC_A_PIN);
  int b = digitalRead(LEG_ENC_B_PIN);

  if (a == b) legEncoderCount++;
  else legEncoderCount--;
}

void IRAM_ATTR hipEncoderISR() {
  int a = digitalRead(HIP_ENC_A_PIN);
  int b = digitalRead(HIP_ENC_B_PIN);

  if (a == b) hipEncoderCount--;
  else hipEncoderCount++;
}

// =====================================================
// Utility
// =====================================================
float legCountsPerDegree() {
  return (float)legEncMax / LEG_MAX_PHYSICAL_DEG;
}

long legDegToEncCounts(float deg) {
  return lroundf(deg * legCountsPerDegree());
}

float legDegPerSecToStepsPerSec(float deg_per_sec) {
  const float STEPS_PER_REV = 400.0f;
  const float STEPS_PER_DEG = STEPS_PER_REV / 360.0f;
  return deg_per_sec * STEPS_PER_DEG;
}

float hipEncCountsToDeg(long counts) {
  return (counts * HIP_DEGREES_PER_1960_COUNTS) / HIP_COUNTS_PER_12_DEG;
}

long hipDegToEncCounts(float deg) {
  return lroundf(deg * HIP_COUNTS_PER_DEG);
}

float hipDegPerSecToStepsPerSec(float deg_per_sec) {
  const float MOTOR_STEPS_PER_DEG = HIP_MOTOR_STEPS_PER_REV / 360.0f;
  return deg_per_sec * HIP_GEAR_RATIO * MOTOR_STEPS_PER_DEG;
}

// =====================================================
// Prepare LEG trajectory
// =====================================================
bool prepareLegTrajectory() {
  for (int i = 0; i < LEG_NPTS; i++) {
    if (leg_pos_wp_deg[i] < 0.0f || leg_pos_wp_deg[i] > LEG_MAX_PHYSICAL_DEG) {
      Serial.println("LEG trajectory angle outside range");
      return false;
    }

    legTargetEncCounts[i] = legDegToEncCounts(leg_pos_wp_deg[i]);
  }

  for (int i = 0; i < LEG_NPTS - 1; i++) {
    float dt = leg_t_wp[i + 1] - leg_t_wp[i];
    float dtheta = leg_pos_wp_deg[i + 1] - leg_pos_wp_deg[i];

    legSegmentSpeedSteps[i] = legDegPerSecToStepsPerSec(dtheta / dt);
  }

  return true;
}

// =====================================================
// Prepare HIP trajectory
// =====================================================
bool prepareHipTrajectory() {
  for (int i = 0; i < HIP_NPTS; i++) {
    if (hip_pos_wp_deg[i] < 0.0f || hip_pos_wp_deg[i] > HIP_MAX_PHYSICAL_DEG) {
      Serial.println("HIP trajectory angle outside range");
      return false;
    }

    hipTargetEncCounts[i] = hipDegToEncCounts(hip_pos_wp_deg[i]);
  }

  for (int i = 0; i < HIP_NPTS - 1; i++) {
    float dt = hip_t_wp[i + 1] - hip_t_wp[i];
    float dtheta = hip_pos_wp_deg[i + 1] - hip_pos_wp_deg[i];

    if (dt <= 0.0f) {
      Serial.println("HIP invalid waypoint timing");
      return false;
    }

    float calculatedSpeed = hipDegPerSecToStepsPerSec(dtheta / dt);

    if (fabs(calculatedSpeed) > HIP_MAX_STEP_SPEED) {
      hipSegmentSpeedSteps[i] = (calculatedSpeed > 0) ? HIP_MAX_STEP_SPEED : -HIP_MAX_STEP_SPEED;
    } else {
      hipSegmentSpeedSteps[i] = calculatedSpeed;
    }
  }

  return true;
}

// =====================================================
// Simultaneous dual-motor feedback move
// =====================================================
void moveBothToWithFeedback(long legTargetEnc, float legSpeedStepsPerSec,
                            long hipTargetEnc, float hipSpeedStepsPerSec) {
  while (true) {
    long legErr = legTargetEnc - legEncoderCount;
    long hipErr = hipTargetEnc - hipEncoderCount;

    bool legDone = (labs(legErr) <= LEG_ENC_TOL_COUNTS);
    bool hipDone = (labs(hipErr) <= HIP_ENC_TOL_COUNTS);

    if (legDone && hipDone) {
      legStepper.setSpeed(0);
      hipStepper.setSpeed(0);
      break;
    }

    if (!legDone) {
      if (legErr > 0) {
        legStepper.setSpeed(LEG_DIR_SIGN * fabs(legSpeedStepsPerSec));
      } else {
        legStepper.setSpeed(-LEG_DIR_SIGN * fabs(legSpeedStepsPerSec));
      }
      legStepper.runSpeed();
    } else {
      legStepper.setSpeed(0);
    }

    if (!hipDone) {
      if (hipErr > 0) {
        hipStepper.setSpeed(HIP_DIR_SIGN * fabs(hipSpeedStepsPerSec));
      } else {
        hipStepper.setSpeed(-HIP_DIR_SIGN * fabs(hipSpeedStepsPerSec));
      }
      hipStepper.runSpeed();
    } else {
      hipStepper.setSpeed(0);
    }
  }
}

// =====================================================
// Run one combined trajectory cycle
// =====================================================
void runCombinedTrajectoryCycle() {
  // Go both to min
  moveBothToWithFeedback(0, LEG_RETURN_SPEED_STEPS,
                         0, HIP_RETURN_SPEED_STEPS);
  delay(50);

  // Go both to first waypoint
  moveBothToWithFeedback(legTargetEncCounts[0], LEG_MOVE_SPEED_STEPS,
                         hipTargetEncCounts[0], HIP_MOVE_SPEED_STEPS);
  delay(50);

  // Timed segments simultaneously
  int maxSegs = (LEG_NPTS - 1 > HIP_NPTS - 1) ? (LEG_NPTS - 1) : (HIP_NPTS - 1);

  for (int i = 0; i < maxSegs; i++) {
    long legTarget = (i < LEG_NPTS - 1) ? legTargetEncCounts[i + 1] : legTargetEncCounts[LEG_NPTS - 1];
    float legSpeed = (i < LEG_NPTS - 1) ? legSegmentSpeedSteps[i] : 0.0f;

    long hipTarget = (i < HIP_NPTS - 1) ? hipTargetEncCounts[i + 1] : hipTargetEncCounts[HIP_NPTS - 1];
    float hipSpeed = (i < HIP_NPTS - 1) ? hipSegmentSpeedSteps[i] : 0.0f;

    moveBothToWithFeedback(legTarget, legSpeed, hipTarget, hipSpeed);
    delay(50);
  }

  delay(200);

  // Return both to min
  moveBothToWithFeedback(0, LEG_RETURN_SPEED_STEPS,
                         0, HIP_RETURN_SPEED_STEPS);
  delay(200);
}

// =====================================================
// Setup
// =====================================================
void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(LEG_ENC_A_PIN, INPUT);
  pinMode(LEG_ENC_B_PIN, INPUT);

  pinMode(HIP_ENC_A_PIN, INPUT);
  pinMode(HIP_ENC_B_PIN, INPUT);

  attachInterrupt(digitalPinToInterrupt(LEG_ENC_A_PIN), legEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(HIP_ENC_A_PIN), hipEncoderISR, CHANGE);

  legStepper.setMaxSpeed(2000);
  hipStepper.setMaxSpeed(5000);

  // Both start physically at MIN = 0
  legEncoderCount = 0;
  legEncMin = 0;

  hipEncoderCount = 0;
  hipEncMin = 0;

  Serial.println("=== COMBINED CALIBRATION ===");
  Serial.println("Both motors are assumed to start at MIN = 0.");
  Serial.println("1) Move LEG to MAX and type 'lm'");
}

// =====================================================
// Loop
// =====================================================
void loop() {
  static unsigned long lastPrint = 0;

  if (millis() - lastPrint > 100) {
    Serial.print("LEG Encoder: ");
    Serial.print(legEncoderCount);
    Serial.print("   HIP Encoder: ");
    Serial.print(hipEncoderCount);
    Serial.print("   HIP Angle(deg): ");
    Serial.println(hipEncCountsToDeg(hipEncoderCount));
    lastPrint = millis();
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (state == WAIT_FOR_LEG_MAX && cmd == "lm") {
      legEncMax = legEncoderCount;
      legMaxCaptured = true;

      Serial.print("LEG MAX captured: ");
      Serial.println(legEncMax);

      Serial.println("2) Move HIP to MAX and type 'hm'");
      state = WAIT_FOR_HIP_MAX;
    }

    else if (state == WAIT_FOR_HIP_MAX && cmd == "hm") {
      hipEncMax = hipEncoderCount;
      hipMaxCaptured = true;

      Serial.print("HIP MAX captured (counts): ");
      Serial.println(hipEncMax);
      Serial.print("HIP MAX captured (deg): ");
      Serial.println(hipEncCountsToDeg(hipEncMax));

      Serial.println("3) Bring BOTH back physically to MIN and type 's'");
      state = WAIT_FOR_START;
    }

    else if (state == WAIT_FOR_START && cmd == "s") {
      // both are now physically back at MIN
      legEncoderCount = 0;
      hipEncoderCount = 0;

      if (!prepareLegTrajectory()) {
        while (1) {}
      }

      if (!prepareHipTrajectory()) {
        while (1) {}
      }

      readyToRun = true;
      digitalWrite(LED_PIN, HIGH);
      Serial.println("Both motors calibrated. Running combined trajectory...");
      state = RUNNING;
    }
  }

  if (state == RUNNING && readyToRun) {
    runCombinedTrajectoryCycle();
  }
}
