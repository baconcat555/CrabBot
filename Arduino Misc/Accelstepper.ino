// Random.pde
// -*- mode: C++ -*-
//
// Make a single stepper perform random changes in speed, position and acceleration
//
// Copyright (C) 2009 Mike McCauley
// $Id: Random.pde,v 1.1 2011/01/05 01:51:01 mikem Exp mikem $
 
#include <AccelStepper.h>
 
// Define a stepper and the pins it will use
AccelStepper stepper3(AccelStepper::DRIVER, 6, 8);
bool direction = 1;
int position = 0;


void setup()
{  
    Serial.begin(9600);
    stepper3.setSpeed(4000);
    stepper3.setAcceleration(200);

}
 
void loop()
{

    if (stepper3.distanceToGo() == 0)
    {
        delay(1000);
        stepper3.moveTo(position);
        stepper3.setMaxSpeed(4000);
        stepper3.setAcceleration(2000);
        position+=400;
    }

    stepper3.run();
    Serial.println("forward");


    // if (stepper3.distanceToGo() == 0)
    // {
    //     delay(1000);
    //     stepper3.moveTo(200);
    //     stepper3.setMaxSpeed(400);
    //     stepper3.setAcceleration(2000);
    // }

    // stepper3.run();
    // Serial.println("backwards");
}

// void loop(){
//     stepper3.moveTo(0);
//     while (stepper3.distanceToGo() > 0){
//         stepper3.runSpeed();
//     }

//     delay(1000);

//     stepper3.moveTo(400);
//     while (stepper3.distanceToGo() > 0){
//         stepper3.runSpeed();
//     }

//     delay(1000);
// }

// void loop()
// {
//     // if (stepper3.distanceToGo() == 0)
//     // {
//     //     // Random change to speed, position and acceleration
//     //     // Make sure we dont get 0 speed or accelerations
//     //     delay(1000);
//     //     stepper3.moveTo(rand() % 2000);
//     //     stepper3.setMaxSpeed((rand() % 2000) + 1);
//     //     stepper3.setAcceleration((rand() % 2000) + 1);
//     // }
//     // stepper3.run();

//     for (int i = 0; i < 400; i++) {
//         digitalWrite(6, HIGH);
//         delay(10);
//         digitalWrite(6, LOW);
//         delay(10);
//     }
//     direction = !direction;
//     digitalWrite(8, direction);
//     delay(1000);
// }
