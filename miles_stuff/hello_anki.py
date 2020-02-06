import anki_vector as av


def main():
    # Modify the SN to match your robot’s SN
    ANKI_SERIAL = '00804458'
    ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

    with av.Robot(serial=ANKI_SERIAL,
                  behavior_control_level=ANKI_BEHAVIOR) as robot:
        print("Say 'Hello World'...")
        robot.behavior.say_text("we're no strangers to love you know the rules and so do i a full commitment's what I'm thinking of you wouldn't get this from any other guy i just want to tell you how I'm feeling, just want to make you understand never going to give you up never going to let you down never going to run around and desert you never going to make you cry never going to say goodbye never going to tell a lie and hurt you")


if __name__ == "__main__":
    main()
