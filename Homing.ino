// Random.pde
// -*- mode: C++ -*-
//
// Make a single stepper perform random changes in speed, position and acceleration
//
// Copyright (C) 2009 Mike McCauley
// $Id: Random.pde,v 1.1 2011/01/05 01:51:01 mikem Exp mikem $
 
#include <AccelStepper.h>


//define global variables:
//for time passed
unsigned long lastMilli = 0;

//If homing then this will be true, start homing
bool toZero = true;

//The position from zero which represents UP.
//number of steps from "0"
int UP = -64;

//max speed without skipping steps
long maxSpeed = 1800;

//Stepper type and pins
AccelStepper stepper(AccelStepper::DRIVER, 6, 8);
 
void interruptFunction() {
  if(toZero){
    stepper.setCurrentPosition(0);
    toZero = false;
  }
}

void autohome(){
    stepper.setAcceleration(1000);
    stepper.setMaxSpeed(maxSpeed);
    stepper.setSpeed(maxSpeed);
    Serial.println("going home");
    toZero = true;
    while(toZero){
        Serial.println("Homing");
        stepper.runSpeed();
    }
    delay(300);
    stepper.moveTo(UP);
    while(stepper.distanceToGo() !=0){
        Serial.println(stepper.currentPosition());
        stepper.runSpeed();
    }
    delay(1000);
}

void setup()
{  
    Serial.begin(9600);
    stepper.setAcceleration(1000);
    stepper.setMaxSpeed(maxSpeed);
    stepper.setSpeed(maxSpeed);
    pinMode(18,INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(18), interruptFunction, CHANGE);
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
    if(millis()-lastMilli > 100){ 
        long currentLeftEncoderCount = stepper.currentPosition();
        Serial.println(currentLeftEncoderCount);
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
}
