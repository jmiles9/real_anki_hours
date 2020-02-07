import anki_vector as av 
import numpy as np
import cv2
from anki_vector.util import degrees

def main():
    ANKI_SERIAL = '00804458'
    ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

    with av.Robot(serial=ANKI_SERIAL,
                  behavior_control_level=ANKI_BEHAVIOR) as robot:
        # initialization sequence - set head angle to 0 and lower fork
        robot.behavior.set_head_angle(degrees(0))
        robot.behavior.set_lift_height(0.0)
        robot.world.connect_cube()
        robot.behavior.say_text("initialization complete")

        # tracking cube
        robot.behavior.say_text("tracking cube")
        cube_img = cv2.imread("block_pattern.jpg", cv2.IMREAD_GRAYSCALE)
        
        # using webcam to get track block (REPLACE THIS WITH ROBOT)
        cv2.VideoCapture(0)  # 0 is webcam number or something
        while(True):
            ret, frame = cap.read()
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', grey)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
