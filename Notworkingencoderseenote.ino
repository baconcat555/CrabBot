// Random.pde
// -*- mode: C++ -*-
//
// Make a single stepper perform random changes in speed, position and acceleration
//
// Copyright (C) 2009 Mike McCauley
// $Id: Random.pde,v 1.1 2011/01/05 01:51:01 mikem Exp mikem $
 
#include <AccelStepper.h>
#include <QuadratureEncoder.h>
//Quadrature encoder.h is the fucking worst get rid of it, only need to 0 based on Z+ from the encoder which is only an interupt
Encoders leftEncoder(2,3);
unsigned long lastMilli = 0;

// Define a stepper and the pins it will use
AccelStepper stepper(AccelStepper::DRIVER, 6, 8);
 
void interruptFunction() {
  stepper.setCurrentPosition(0);
}

void setup()
{  
    Serial.begin(9600);
    pinMode(4, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(4), interruptFunction, CHANGE);
}
 
void loop()
{
    if (stepper.distanceToGo() == 0)
    {
        // Random change to speed, position and acceleration
        // Make sure we dont get 0 speed or accelerations
        delay(1000);
        stepper.moveTo(rand() % 1600);
        stepper.setMaxSpeed((rand() % 1800) + 50);
        stepper.setAcceleration((rand() % 200) + 30);
    }


    if(millis()-lastMilli > 50){ 
        long currentLeftEncoderCount = leftEncoder.getEncoderCount();
        Serial.println(currentLeftEncoderCount);
        lastMilli = millis();
  }

    stepper.run();
}
