#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

void setup()
{
  Serial.begin(9600);

  Braccio.begin();

  // Initial safe position
  Braccio.ServoMovement(
    20,
    90,   // base
    90,   // shoulder
    90,   // elbow
    90,   // wrist rotation
    90,   // wrist vertical
    10    // gripper
  );
}

void loop()
{
  if (Serial.available() > 0)
  {
    String command = Serial.readStringUntil('\n');

    int baseAngle;
    int shoulderAngle;
    int elbowAngle;
    int wristRotAngle;
    int wristVerAngle;
    int gripperAngle;

    int result = sscanf(
      command.c_str(),
      "%d,%d,%d,%d,%d,%d",
      &baseAngle,
      &shoulderAngle,
      &elbowAngle,
      &wristRotAngle,
      &wristVerAngle,
      &gripperAngle
    );

    if (result == 6)
    {
      Braccio.ServoMovement(
        20,
        baseAngle,
        shoulderAngle,
        elbowAngle,
        wristRotAngle,
        wristVerAngle,
        gripperAngle
      );

      Serial.println("OK");
    }
  }
}