import anki_vector as av 
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



if __name__ == "__main__":
    main()
