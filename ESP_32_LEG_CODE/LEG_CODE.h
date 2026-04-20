
#include <AccelStepper.h>
#include <limits.h>
#include "BluetoothSerial.h"
BluetoothSerial ESP_BT;

// =====================================================
// ESP_BT COMMANDS
// 1 = run trajectory slow
// 2 = run trajectory medium
// 3 = run trajectory fast
// x = stop
// h = help
// =====================================================

// =====================================================
// MOTOR / ENCODER PINS
// =====================================================
#define LEG_STEP_PIN   5
#define LEG_DIR_PIN    4
#define LEG_ENC_A_PIN  19
#define LEG_ENC_B_PIN  18

#define HIP_STEP_PIN   32
#define HIP_DIR_PIN    33
#define HIP_ENC_A_PIN  35
#define HIP_ENC_B_PIN  34

#define LED_PIN        2

// =====================================================
// DIRECTION SIGNS
// Keep these consistent with your verified direction test
// =====================================================
#define LEG_DIR_SIGN   -1
#define HIP_DIR_SIGN    1

// =====================================================
// LEG GEOMETRY
// Startup pose at power-on = 0 deg
//
// If your encoder is 1960 counts / rev on the same shaft:
// 1960 / 360 = 5.444 counts/deg
// =====================================================
const float LEG_MAX_PHYSICAL_DEG = 166.62f;
const float LEG_COUNTS_PER_DEG   = 1960.0f / 360.0f;
const long  LEG_ENC_TOL_COUNTS   = 3;

// =====================================================
// HIP GEOMETRY
// You explicitly measured:
// 1960 encoder counts = 12 degrees
// =====================================================
const float HIP_MAX_PHYSICAL_DEG       = 60.0f;
const float HIP_COUNTS_PER_12_DEG      = 1960.0f;
const float HIP_DEGREES_PER_1960_COUNTS = 12.0f;
const float HIP_COUNTS_PER_DEG         = HIP_COUNTS_PER_12_DEG / HIP_DEGREES_PER_1960_COUNTS;
const long  HIP_ENC_TOL_COUNTS         = 10;

// =====================================================
// TRAJECTORY
// Full trajectory points in degrees
// =====================================================
const int LEG_NPTS = 8;
float leg_pos_wp_deg[LEG_NPTS] = {
  3.91f, 30.58f, 61.17f, 95.78f,
  158.15f, 128.15f, 3.91f, 3.91f
};
float leg_t_wp[LEG_NPTS] = {
  0.000f, 0.232f, 0.463f, 0.694f,
  0.926f, 1.078f, 1.542f, 1.600f
};

const int HIP_NPTS = 8;
float hip_pos_wp_deg[HIP_NPTS] = {
  3.55f, 5.45f, 5.54f, 3.89f,
  0.40f, -7.27f, -9.51f, 3.55f
};
float hip_t_wp[HIP_NPTS] = {
  0.000f, 0.232f, 0.463f, 0.694f,
  0.926f, 1.078f, 1.542f, 1.600f
};

long legTargetFwd[LEG_NPTS];
long hipTargetFwd[HIP_NPTS];

// =====================================================
// RUN MODES
// =====================================================
enum RunMode {
  RUNMODE_STOP,
  RUNMODE_SLOW,
  RUNMODE_MED,
  RUNMODE_FAST
};

RunMode runMode = RUNMODE_STOP;

// =====================================================
// PID GAINS
// Start with P only, tune I and D later
// =====================================================
float LEG_KP = 8.8f;
float LEG_KI = 0.8f;
float LEG_KD = 0.0f;

float HIP_KP = 0.8f;
float HIP_KI = 0.0f;
float HIP_KD = 0.0f;

// =====================================================
// PID LIMITS / COMMAND LIMITS
// =====================================================
float LEG_I_LIMIT = 300.0f;
float HIP_I_LIMIT = 300.0f;

// PID correction ceiling
float LEG_RECOVERY_MAX_SPEED = 180.0f;
float HIP_RECOVERY_MAX_SPEED = 120.0f;

// Absolute final command ceiling
float LEG_ABS_MAX_SPEED = 1020.0f;
float HIP_ABS_MAX_SPEED = 1020.0f;

// Minimum useful drive to overcome stiction
float LEG_MIN_CMD_SPEED = 35.0f;
float HIP_MIN_CMD_SPEED = 35.0f;

// =====================================================
// CONTROL LOOP TIMING
// =====================================================
const unsigned long PID_DT_US   = 2000;   // 2 ms
const unsigned long DEBUG_DT_MS = 150;

// =====================================================
// SPEED PROFILES
// These define nominal trajectory speed only
// =====================================================
float speedScale() {
  switch (runMode) {
    case RUNMODE_SLOW: return 0.65f;
    case RUNMODE_MED:  return 0.85f;
    case RUNMODE_FAST: return 1.00f;
    default:           return 1.0f;
  }
}

float currentLegReturnSpeed() {
  switch (runMode) {
    case RUNMODE_SLOW: return 620.0f;
    case RUNMODE_MED:  return 850.0f;
    case RUNMODE_FAST: return 980.0f;
    default:           return 400.0f;
  }
}

float currentHipReturnSpeed() {
  switch (runMode) {
    case RUNMODE_SLOW: return 620.0f;
    case RUNMODE_MED:  return 850.0f;
    case RUNMODE_FAST: return 980.0f;
    default:           return 400.0f;
  }
}

// =====================================================
// STEPPERS + ENCODERS
// =====================================================
AccelStepper legStepper(AccelStepper::DRIVER, LEG_STEP_PIN, LEG_DIR_PIN);
AccelStepper hipStepper(AccelStepper::DRIVER, HIP_STEP_PIN, HIP_DIR_PIN);

volatile long legEncoderCount = 0;
volatile long hipEncoderCount = 0;

// startup reference = zero
long legStartupOffset = 0;
long hipStartupOffset = 0;

// PID state
float legIntegral = 0.0f;
float hipIntegral = 0.0f;
float legPrevErr  = 0.0f;
float hipPrevErr  = 0.0f;

// =====================================================
// ENCODER ISRs
// Keep these consistent with your verified manual test
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

  if (a == b) hipEncoderCount++;
  else hipEncoderCount--;
}

// =====================================================
// HELPERS
// =====================================================
float clampFloat(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

long legPos() {
  return legEncoderCount - legStartupOffset;
}

long hipPos() {
  return hipEncoderCount - hipStartupOffset;
}

long legDegToEnc(float deg) {
  return lroundf(deg * LEG_COUNTS_PER_DEG);
}

long hipDegToEnc(float deg) {
  return lroundf(deg * HIP_COUNTS_PER_DEG);
}

void resetPIDStates() {
  legIntegral = 0.0f;
  hipIntegral = 0.0f;
  legPrevErr  = 0.0f;
  hipPrevErr  = 0.0f;
}

// =====================================================
// BUILD TRAJECTORY
// =====================================================
void buildTrajectory() {
  for (int i = 0; i < LEG_NPTS; i++) {
    legTargetFwd[i] = legDegToEnc(leg_pos_wp_deg[i]);
  }

  for (int i = 0; i < HIP_NPTS; i++) {
    hipTargetFwd[i] = hipDegToEnc(hip_pos_wp_deg[i]);
  }
}

// =====================================================
// PID RECOVERY CORRECTION
// correction is added to nominal segment speed
// =====================================================
float computeRecoveryCorrection(long err,
                                float dt,
                                float &integral,
                                float &prevErr,
                                float kp,
                                float ki,
                                float kd,
                                float iLimit,
                                float recoveryMaxSpeed,
                                long tolCounts,
                                float minCmdSpeed) {
  if (labs(err) <= tolCounts) {
    integral = 0.0f;
    prevErr = (float)err;
    return 0.0f;
  }

  integral += ((float)err) * dt;
  integral = clampFloat(integral, -iLimit, iLimit);

  float derr = (((float)err) - prevErr) / dt;
  float correction = kp * ((float)err) + ki * integral + kd * derr;
  prevErr = (float)err;

  correction = clampFloat(correction, -recoveryMaxSpeed, recoveryMaxSpeed);

  if (fabs(correction) > 0.0f && fabs(correction) < minCmdSpeed) {
    correction = (correction > 0.0f) ? minCmdSpeed : -minCmdSpeed;
  }

  return correction;
}

// =====================================================
// ESP_BT HELP
// =====================================================
void printHelp() {
  ESP_BT.println();
  ESP_BT.println("=== COMMANDS ===");
  ESP_BT.println("1 = run trajectory slow");
  ESP_BT.println("2 = run trajectory medium");
  ESP_BT.println("3 = run trajectory fast");
  ESP_BT.println("x = stop");
  ESP_BT.println("h = help");
  ESP_BT.println("================");
  ESP_BT.println();
}

void applyESP_BTCommand(char c) {
  switch (c) {
    case '1':
      runMode = RUNMODE_SLOW;
      ESP_BT.println("Mode: SLOW");
      break;
    case '2':
      runMode = RUNMODE_MED;
      ESP_BT.println("Mode: MED");
      break;
    case '3':
      runMode = RUNMODE_FAST;
      ESP_BT.println("Mode: FAST");
      break;
    case 'x':
    case 'X':
      runMode = RUNMODE_STOP;
      ESP_BT.println("Mode: STOP");
      break;
    case 'h':
    case 'H':
      printHelp();
      break;
    default:
      break;
  }
}

// =====================================================
// MOVE BOTH MOTORS TO TARGET
// final speed = nominal segment speed + PID correction
// =====================================================
void moveBothTo(long legTargetEnc, float legNominalSpeed,
                long hipTargetEnc, float hipNominalSpeed) {
  static unsigned long lastDebugMs = 0;
  unsigned long lastPidUs = micros();

  resetPIDStates();

  while (true) {
    if (ESP_BT.available()) {
      char c = ESP_BT.read();
      applyESP_BTCommand(c);
    }

    if (runMode == RUNMODE_STOP) {
      legStepper.setSpeed(0);
      hipStepper.setSpeed(0);
      return;
    }

    long legNow = legPos();
    long hipNow = hipPos();

    long legErr = legTargetEnc - legNow;
    long hipErr = hipTargetEnc - hipNow;

    bool legDone = labs(legErr) <= LEG_ENC_TOL_COUNTS;
    bool hipDone = labs(hipErr) <= HIP_ENC_TOL_COUNTS;

    if (legDone && hipDone) {
      legStepper.setSpeed(0);
      hipStepper.setSpeed(0);
      return;
    }

    unsigned long nowUs = micros();
    unsigned long dtUs = nowUs - lastPidUs;

    if (dtUs >= PID_DT_US) {
      float dt = dtUs * 1.0e-6f;
      lastPidUs = nowUs;

      if (!legDone) {
        float legCorrection = computeRecoveryCorrection(
          legErr, dt,
          legIntegral, legPrevErr,
          LEG_KP, LEG_KI, LEG_KD,
          LEG_I_LIMIT,
          LEG_RECOVERY_MAX_SPEED,
          LEG_ENC_TOL_COUNTS,
          LEG_MIN_CMD_SPEED
        );

        float legBase = (legErr > 0) ? fabs(legNominalSpeed) : -fabs(legNominalSpeed);
        float legCmd  = legBase + legCorrection;
        legCmd = clampFloat(legCmd, -LEG_ABS_MAX_SPEED, LEG_ABS_MAX_SPEED);

        legStepper.setSpeed(LEG_DIR_SIGN * legCmd);
      } else {
        legStepper.setSpeed(0);
      }

      if (!hipDone) {
        float hipCorrection = computeRecoveryCorrection(
          hipErr, dt,
          hipIntegral, hipPrevErr,
          HIP_KP, HIP_KI, HIP_KD,
          HIP_I_LIMIT,
          HIP_RECOVERY_MAX_SPEED,
          HIP_ENC_TOL_COUNTS,
          HIP_MIN_CMD_SPEED
        );

        float hipBase = (hipErr > 0) ? fabs(hipNominalSpeed) : -fabs(hipNominalSpeed);
        float hipCmd  = hipBase + hipCorrection;
        hipCmd = clampFloat(hipCmd, -HIP_ABS_MAX_SPEED, HIP_ABS_MAX_SPEED);

        hipStepper.setSpeed(HIP_DIR_SIGN * hipCmd);
      } else {
        hipStepper.setSpeed(0);
      }
    }

    legStepper.runSpeed();
    hipStepper.runSpeed();

    if (millis() - lastDebugMs > DEBUG_DT_MS) {
      ESP_BT.print("LEG target=");
      ESP_BT.print(legTargetEnc);
      ESP_BT.print(" pos=");
      ESP_BT.print(legNow);
      ESP_BT.print(" err=");
      ESP_BT.print(legErr);

      ESP_BT.print(" | HIP target=");
      ESP_BT.print(hipTargetEnc);
      ESP_BT.print(" pos=");
      ESP_BT.print(hipNow);
      ESP_BT.print(" err=");
      ESP_BT.println(hipErr);

      lastDebugMs = millis();
    }
  }
}

// =====================================================
// RETURN TO STARTUP ZERO
// =====================================================
void goHomeAndRest() {
  moveBothTo(0, currentLegReturnSpeed(), 0, currentHipReturnSpeed());
}

// =====================================================
// FULL TRAJECTORY IMPLEMENTATION
// =====================================================
void runWalkCycle() {
  float s = speedScale();

  // first point slower
  moveBothTo(legTargetFwd[0], 60.0f * s, hipTargetFwd[0], 60.0f * s);
  if (runMode == RUNMODE_STOP) return;

  int maxSegs = max(LEG_NPTS - 1, HIP_NPTS - 1);

  for (int i = 0; i < maxSegs; i++) {
    long lT = (i < LEG_NPTS - 1) ? legTargetFwd[i + 1] : legTargetFwd[LEG_NPTS - 1];
    long hT = (i < HIP_NPTS - 1) ? hipTargetFwd[i + 1] : hipTargetFwd[HIP_NPTS - 1];

    float lNom;
    if (i < LEG_NPTS - 1) {
      float dt = leg_t_wp[i + 1] - leg_t_wp[i];
      lNom = fabsf((legTargetFwd[i + 1] - legTargetFwd[i]) / dt) * s;
    } else {
      lNom = currentLegReturnSpeed();
    }

    float hNom;
    if (i < HIP_NPTS - 1) {
      float dt = hip_t_wp[i + 1] - hip_t_wp[i];
      hNom = fabsf((hipTargetFwd[i + 1] - hipTargetFwd[i]) / dt) * s;
    } else {
      hNom = currentHipReturnSpeed();
    }

    // keep the last few leg segments gentler
    if (i >= LEG_NPTS - 3) {
      lNom *= 0.6f;
    }

    // clamp nominal speeds
    if (lNom > 120.0f) lNom = 120.0f;
    if (hNom > 150.0f) hNom = 150.0f;

    moveBothTo(lT, lNom, hT, hNom);
    if (runMode == RUNMODE_STOP) return;
  }
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  ESP_BT.begin("ESP32_LEG_3");


  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  pinMode(LEG_ENC_A_PIN, INPUT_PULLUP);
  pinMode(LEG_ENC_B_PIN, INPUT_PULLUP);

  // GPIO34/35 are input-only and do not support internal pullups
  pinMode(HIP_ENC_A_PIN, INPUT);
  pinMode(HIP_ENC_B_PIN, INPUT);

  attachInterrupt(digitalPinToInterrupt(LEG_ENC_A_PIN), legEncoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(HIP_ENC_A_PIN), hipEncoderISR, CHANGE);

  delay(50);

  // power-up pose becomes zero reference
  legStartupOffset = legEncoderCount;
  hipStartupOffset = hipEncoderCount;

  legStepper.setMaxSpeed(2000);
  hipStepper.setMaxSpeed(2000);
  legStepper.setMinPulseWidth(5);
  hipStepper.setMinPulseWidth(5);

  buildTrajectory();

  ESP_BT.println("Full 2-motor PID trajectory controller ready.");
  ESP_BT.println("Startup pose treated as zero for both axes.");
  ESP_BT.print("LEG_COUNTS_PER_DEG = ");
  ESP_BT.println(LEG_COUNTS_PER_DEG);
  ESP_BT.print("HIP_COUNTS_PER_DEG = ");
  ESP_BT.println(HIP_COUNTS_PER_DEG);

  ESP_BT.println("Leg Targets:");
  for (int i = 0; i < LEG_NPTS; i++) {
    ESP_BT.print("legTargetFwd[");
    ESP_BT.print(i);
    ESP_BT.print("] = ");
    ESP_BT.println(legTargetFwd[i]);
  }

  ESP_BT.println("Hip Targets:");
  for (int i = 0; i < HIP_NPTS; i++) {
    ESP_BT.print("hipTargetFwd[");
    ESP_BT.print(i);
    ESP_BT.print("] = ");
    ESP_BT.println(hipTargetFwd[i]);
  }

  printHelp();
}

// =====================================================
// LOOP
// =====================================================
void loop() {
  static unsigned long lastPrintMs = 0;

  if (ESP_BT.available()) {
    char c = ESP_BT.read();
    applyESP_BTCommand(c);
  }

  switch (runMode) {
    case RUNMODE_STOP:
      goHomeAndRest();
      break;

    case RUNMODE_SLOW:
    case RUNMODE_MED:
    case RUNMODE_FAST:
      runWalkCycle();
      break;
  }

  if (millis() - lastPrintMs > 200) {
    ESP_BT.print("LEG pos=");
    ESP_BT.print(legPos());
    ESP_BT.print(" | HIP pos=");
    ESP_BT.print(hipPos());
    ESP_BT.print(" | mode=");
    ESP_BT.println((int)runMode);
    lastPrintMs = millis();
  }
}
