#! /usr/bin/env python

# this should listen to the camera feed and publish to the /cmd_vel
# listen to topic /robot/camera1/image_raw

import rospy
import cv2
from sensor_msgs.msg import Image  # this is the thing the camera will publish
from cv_bridge import CvBridge, CvBridgeError


# define a callback function which will handle incoming messages
# data = a frame from the camera feed
def callback(data):
    # translate to cv2 format
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    # gets the segmented versions of the image
    plate_segments = # TODO

    # steer toward cm
    new_vels = Twist()
    new_vels.linear.x = 1

    height, width, chan = cv_image.shape
    centre = width/2
    thval = 5
    if cm < centre - thval:
        # turn left
        new_vels.angular.z = 1
    elif cm > centre + thval:
        # turn right
        new_vels.angular.z = -1

    pub.publish(new_vels)


def calculate_cm(frame):
    height, width, chan = frame.shape
    subframe = frame[height-100:height, 0:width]

    grey = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
    threshold = 90
    _, thresh = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY_INV)

    _, contour_list, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cX = 0
    if len(contour_list) != 0:
        c = max(contour_list, key=cv2.contourArea)
        moms = cv2.moments(c)
        if moms["m00"] != 0:
            cX = int(moms["m10"]/moms["m00"])

    return cX


bridge = CvBridge()
rospy.init_node('line_follow')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
rospy.Subscriber('/robot/camera1/image_raw', Image, callback)

rospy.spin()
