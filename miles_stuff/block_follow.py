import anki_vector as av 
import numpy as np
import cv2
from anki_vector.util import degrees

def main():
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

    with av.Robot(serial=ANKI_SERIAL,
                  behavior_control_level=ANKI_BEHAVIOR) as robot:
        # initialization sequence - set head angle to 0 and lower fork
        robot.behavior.set_head_angle(degrees(0))
        robot.behavior.set_lift_height(0.0)
        robot.world.connect_cube()
        robot.behavior.say_text("initialization complete")

        # tracking cube
        robot.behavior.say_text("tracking cube")
        robot.camera.init_camera_feed()
         
        # get video feed from anki (frame by frame), convert to openCV format
        while(True):
            raw_img = robot.camera.latest_image.raw_image
            frame = np.array(raw_img)




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

        # compare key points
        if len(good_points) >= 4:
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # perspective transform
            h, w = cube_img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            if matrix is not None:
                dst = cv2.perspectiveTransform(pts, matrix)

                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), cv2.LINE_AA)

                cv2.imshow('frame', homography)
                robot.behavior.say_text("cube found!")
            else:
                cv2.imshow('frame', frame)
        else:
            cv2.imshow('frame', frame)
            robot.behavior.say_text("aaaaaa")
    
    return "this should be the centrepoint of cube or something"


if __name__ == "__main__":
    main()
