#include <Servo.h>


Servo panServo;
Servo tiltServo;

int pan = 90;
int tilt = 100;

void setup() {
  Serial.begin(115200);

  panServo.attach(9);
  tiltServo.attach(10);

  panServo.write(pan);
  tiltServo.write(tilt);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');

    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      pan = data.substring(0, commaIndex).toFloat();
      tilt = data.substring(commaIndex + 1).toFloat();

      pan = constrain(pan, 0, 180);
      tilt = constrain(tilt, 0, 180);
      Serial.print("Recieved ");
      Serial.print(pan);
      Serial.print(" , ");
      Serial.println(tilt);
      

      panServo.write(pan);
      tiltServo.write(tilt);
    }
  }
}
