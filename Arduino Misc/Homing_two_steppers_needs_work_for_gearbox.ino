// Stepper Homing
//
// Homes 2 stepper motors to a "home" point defined by a quadrature encoder which pulses once per revolution
//
// uses code from Random.pde ((C) 2009 Mike McCauley)
 
#include <AccelStepper.h>


//define global variables:
//for time passed
unsigned long lastMilli = 0;

//If homing then this will be true, start homing
bool toZero1 = true;
bool toZero2 = true;

//The position from zero which represents UP.
//number of steps from "0"
int UP = -64;

//max speed without skipping steps
long maxSpeed = 1800;

//Stepper type and pins
AccelStepper stepper(AccelStepper::DRIVER, 6,7);
AccelStepper stepper2(AccelStepper::DRIVER, 8,9);
 
void interruptFunction() {
Serial.println("Stepper1");
  if(toZero1){
    stepper.setCurrentPosition(0);
    toZero1 = false;
  }
}

void interruptFunction2() {
Serial.println("Stepper2");
if(toZero2){
    stepper2.setCurrentPosition(0);
    toZero2 = false;
}
}


void autohome(){
    stepper.setAcceleration(1000);
    stepper.setMaxSpeed(maxSpeed);
    stepper.setSpeed(maxSpeed);
    //Serial.println("going home");
    toZero1 = true;
    toZero2 = true;
    while(toZero1){
        stepper.runSpeed();
    }
    while(toZero2){
        stepper2.runSpeed();
    }
    delay(300);
    stepper.moveTo(UP);
    stepper2.moveTo(UP);

    while(stepper.distanceToGo() !=0 || stepper2.distanceToGo() !=0){
        if(stepper.distanceToGo() !=0){
            stepper.runSpeed();
        }
        if(stepper2.distanceToGo() !=0){
            stepper2.runSpeed();
        }
    }
    delay(1000);
}

void setup()
{  
    Serial.begin(9600);
    stepper.setAcceleration(1000);
    stepper.setMaxSpeed(maxSpeed);
    stepper.setSpeed(maxSpeed);

    stepper2.setAcceleration(1000);
    stepper2.setMaxSpeed(maxSpeed);
    stepper2.setSpeed(maxSpeed);

    pinMode(2,INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(2), interruptFunction, CHANGE);
    pinMode(3,INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(3), interruptFunction2, CHANGE);
    autohome();
}
 

void loop()
{
    if (stepper.distanceToGo() == 0)
    {
        delay(10);
        stepper.moveTo(rand() % 3200-1600);
        stepper.setMaxSpeed((rand() % 1800) + 100);
        stepper.setAcceleration((rand() % 200) + 30);
    }

    if (stepper2.distanceToGo() == 0)
    {
        delay(10);
        stepper2.moveTo(rand() % 3200-1600);
        stepper2.setMaxSpeed((rand() % 1800) + 100);
        stepper2.setAcceleration((rand() % 200) + 30);
    }

    if(millis()-lastMilli > 100){ 
        long currentLeftEncoderCount = stepper.currentPosition();
        long currentLeftEncoderCount2 = stepper2.currentPosition();
        //Serial.println(currentLeftEncoderCount);
        //Serial.println(currentLeftEncoderCount2);
        lastMilli = millis();
  }
    if(Serial.available()>0){
        String text = Serial.readString();
        text.trim();
        if(!text.compareTo("1")){
            Serial.println("Going home");
            autohome();
        }
    }
    
    stepper.run();
    stepper2.run();
}
