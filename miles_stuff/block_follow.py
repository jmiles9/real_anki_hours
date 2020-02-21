#!/usr/bin/python3
# above line tells unix what interpreter to use (since have both 2 and 3)

import anki_vector as av 
import numpy as np
import math
import cv2
from anki_vector.util import degrees

def main():

    def findCube(frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_grey, dsc_grey = sift.detectAndCompute(grey, None) #keypoints and descriptors
        matches = flann.knnMatch(dsc_cubeimg, dsc_grey, k=2)  # get matches using flann btwn image and frame
        good_points = []
        # above finds k matches for each descriptor. then compares to see which is better ?
        for m, n in matches:
            if m.distance < 0.6*n.distance:  # why 0.6???
                good_points.append(m)

        # get key points from each source into nice format 
        query_pts = np.float32([kp_cubeimg[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grey[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

        dst = None
        # compare key points
        if len(good_points) >= 4:
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # perspective transform
            h, w = cube_img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            if matrix is not None:
                dst = cv2.perspectiveTransform(pts, matrix)
                cd1 = (dst[0][0][0], dst[0][0][1])
                # cd2 = (dst[1][0][0], dst[1][0][1])
                # cd3 = (dst[2][0][0], dst[2][0][1])
                # cd4 = (dst[3][0][0], dst[3][0][1])

                # homography = cv2.polylines(grey, [np.int32(dst)], True, (255, 0, 0), cv2.LINE_AA)
                cv2.circle(grey, cd1, 5, (0, 255, 0), -1)
                # cv2.circle(grey, cd2, 5, (0, 0, 255), -1)
                # cv2.circle(grey, cd3, 5, (0, 255, 255), -1)
                # cv2.circle(grey, cd4, 5, (255, 255, 0), -1)
                cv2.imshow('frame', grey)       
                
            else:
                cv2.imshow('frame', grey)
                # print("else1")
        else:
            cv2.imshow('frame', grey)
            # print("else2")
        
        cv2.waitKey(1)
        return dst  # dst_tup is the top right (?) corner of cube position

    def calculateDistance(dst_matrix, frame):
        # corners of square
        cd1 = (dst_matrix[0][0][0], dst_matrix[0][0][1])
        cd2 = (dst_matrix[1][0][0], dst_matrix[1][0][1])
        cd3 = (dst_matrix[2][0][0], dst_matrix[2][0][1])
        cd4 = (dst_matrix[3][0][0], dst_matrix[3][0][1])

        # if it is to the left or right
        w_sum = 0
        h_sum = 0
        h_list = []
        for i in range(4):
            h_list.append(dst_matrix[i][0][1])
            w_sum += dst_matrix[i][0][0]
            h_sum += h_list[i]

        centre = (w_sum/4, h_sum/4)

        # how far away it is
        h_list.sort() # smallest to largest

        h_len = 0.5*(h_list[3] + h_list[2]) - 0.5*(h_list[1] + h_list[0])
        h_frame, w_frame, _ = frame.shape 

        raw_dist = abs(h_frame - h_len)

        if raw_dist < 310:
            d_from_cube = 20 - math.sqrt((310-raw_dist)/0.4)
        else:
            d_from_cube = 20  # maybe something else here TODO

        centre_dist = w_frame/2 - w_sum/4
        # print(f"raw_dist = {raw_dist}")
        # print(f"d_from_cube = {d_from_cube}")
        return d_from_cube, centre_dist


        
    ANKI_SERIAL = '00804458'
    ANKI_BEHAVIOR = av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY
    cube_img = cv2.imread("block_pattern.jpg", cv2.IMREAD_GRAYSCALE)

    # sift algorythm
    sift = cv2.xfeatures2d.SIFT_create()
    kp_cubeimg, dsc_cubeimg = sift.detectAndCompute(cube_img, None)

    # feature matching algorythm
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()  # empty dictionary ?
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # other
    SET_POINT = 10  # desired distance from cube, cm

    with av.Robot(serial=ANKI_SERIAL,
                  behavior_control_level=ANKI_BEHAVIOR) as robot:
        # initialization sequence - set head angle to 0 and lower fork
        robot.behavior.set_head_angle(degrees(0))
        robot.behavior.set_lift_height(0.0)
        # robot.world.connect_cube()
        robot.behavior.say_text("initialization complete")

        # tracking cube
        robot.camera.init_camera_feed()
        robot.behavior.say_text("tracking cube")
         
        # get video feed from anki (frame by frame), convert to openCV format
        while(True):
            # print(robot.camera.image_streaming_enabled())
            img = robot.camera.latest_image
            raw_img = img.raw_image
            frame_cv2 = np.array(raw_img)

            pos = findCube(frame_cv2)
            if pos is not None:
                d, c_error = calculateDistance(pos, frame_cv2)
                d_error = d - SET_POINT

            else:
                # if it doesn't see the cube, do nothing
                # TODO maybe change this to look around?
                d_error = 0
                c_error = 0
            
            # move robot!!
            # if its different than 10cm away, move forward or backward
            # and if off centre, turn
            l_w_speed = d_error*10 - c_error/7
            r_w_speed = d_error*10 + c_error/7
            robot.motors.set_wheel_motors(l_w_speed, r_w_speed)
            print(r_w_speed)



if __name__ == "__main__":
    main()
