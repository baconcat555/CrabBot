#include <AccelStepper.h>

// -------------------------
// Pin definitions
// -------------------------
const int STEP_PIN = 6;
const int DIR_PIN  = 7;
const int HOME_Z_PIN = 2;   // Z/index pulse input after receiver

// -------------------------
// Stepper object
// DRIVER mode = STEP + DIR
// -------------------------
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// -------------------------
// User settings
// -------------------------
const float CONSTANT_SPEED = 75.0;     
const long STEPS_PER_REV = 400;        

// Set your 3 desired absolute angles here
float angle1_deg = -45.0;
float angle2_deg = -60.0;
float angle3_deg = -75.0;

// Homing direction
const float HOME_SPEED = -200.0;

// -------------------------
// Homing flag set by interrupt
// -------------------------
volatile bool homeDetected = false;

// -------------------------
// Interrupt for Z pulse
// -------------------------
void homeISR() {
  homeDetected = true;
}

// -------------------------
// Convert angle to motor steps
// -------------------------
long angleToSteps(float angleDeg) {
  return (long)((angleDeg / 360.0) * STEPS_PER_REV);
}

// -------------------------
// Home motor using Z pulse
// -------------------------
void autoHome() {
  Serial.println("Homing started...");
  homeDetected = false;

  stepper.setSpeed(HOME_SPEED);

  while (!homeDetected) {
    stepper.runSpeed();
  }

  stepper.setCurrentPosition(0);
  Serial.println("Home found. Position set to 0.");
}

// -------------------------
// Move relative number of steps at constant speed
// -------------------------
void moveRelativeConstantSpeed(long relativeSteps) {
  stepper.move(relativeSteps);

  if (relativeSteps >= 0) {
    stepper.setSpeed(CONSTANT_SPEED);
  } else {
    stepper.setSpeed(-CONSTANT_SPEED);
  }

  while (stepper.distanceToGo() != 0) {
    stepper.runSpeedToPosition();
  }
}

// -------------------------
// Move to absolute target in steps at constant speed
// -------------------------
void moveAbsoluteConstantSpeed(long absoluteSteps) {
  long delta = absoluteSteps - stepper.currentPosition();
  moveRelativeConstantSpeed(delta);
}

// -------------------------
// Compute shortest move back to home (0 position)
// considering wrap-around over one revolution
// -------------------------
long shortestDeltaToHome(long currentPos) {
  long modPos = currentPos % STEPS_PER_REV;
  if (modPos < 0) {
    modPos += STEPS_PER_REV;
  }

  long delta = -modPos;  // move back toward 0

  if (delta < -(STEPS_PER_REV / 2)) {
    delta += STEPS_PER_REV;
  } else if (delta > (STEPS_PER_REV / 2)) {
    delta -= STEPS_PER_REV;
  }

  return delta;
}

// -------------------------
// Run full sequence once
// -------------------------
void runSequence() {
  // 1. Home
  autoHome();

  // 2. Stay at home for 2 seconds
  Serial.println("Waiting at home for 2 seconds");
  delay(2000);

  // Convert angles to absolute motor step positions
  long pos1_steps = angleToSteps(angle1_deg);
  long pos2_steps = angleToSteps(angle2_deg);
  long pos3_steps = angleToSteps(angle3_deg);

  Serial.print("Angle 1 steps: "); Serial.println(pos1_steps);
  Serial.print("Angle 2 steps: "); Serial.println(pos2_steps);
  Serial.print("Angle 3 steps: "); Serial.println(pos3_steps);

  // 3. Go to first angle from home
  Serial.println("Going to position 1...");
  moveAbsoluteConstantSpeed(pos1_steps);
  Serial.print("Reached position 1, current step position = ");
  Serial.println(stepper.currentPosition());
  delay(1000);

  // 4. Go from angle 1 to angle 2 using difference
  Serial.println("Going to position 2...");
  long diff12 = pos2_steps - pos1_steps;
  moveRelativeConstantSpeed(diff12);
  Serial.print("Reached position 2, current step position = ");
  Serial.println(stepper.currentPosition());
  delay(1000);

  // 5. Go from angle 2 to angle 3 using difference
  Serial.println("Going to position 3...");
  long diff23 = pos3_steps - pos2_steps;
  moveRelativeConstantSpeed(diff23);
  Serial.print("Reached position 3, current step position = ");
  Serial.println(stepper.currentPosition());
  delay(1000);

  // 6. Return to home using shortest rotation direction
  Serial.println("Returning to home using shortest path...");
  long deltaHome = shortestDeltaToHome(stepper.currentPosition());
  moveRelativeConstantSpeed(deltaHome);
  stepper.setCurrentPosition(0);

  Serial.println("Sequence complete.");
}

void setup() {
  Serial.begin(115200);

  // Stepper setup
  stepper.setMaxSpeed(CONSTANT_SPEED);
  stepper.setSpeed(CONSTANT_SPEED);

  // Home input
  pinMode(HOME_Z_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(HOME_Z_PIN), homeISR, RISING);

  // Run once
  runSequence();
}

void loop() {
  // Empty for now
}
