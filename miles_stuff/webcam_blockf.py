import numpy as np
import cv2


def main():
    cube_img = cv2.imread("block_pattern.jpg", cv2.IMREAD_GRAYSCALE)
        
    # using webcam to get track block (REPLACE THIS WITH ROBOT)
    cap = cv2.VideoCapture(0)  # 0 is webcam number or something
    # get sift algorythm
    sift = cv2.xfeatures2d.SIFT_create()
    kp_cubeimg, dsc_cubeimg = sift.detectAndCompute(cube_img, None)

    # load the feature matching algorythm (what is trees???)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()  # empty dictionary ?
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # detect features of image in webcam image
    while(True):
        ret, frame = cap.read()
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
        #for i in train_pts:
            #tup = (i)

        # compare key points
        if len(good_points) >= 4:
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # perspective transform
            h, w = cube_img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            if matrix is not None:
                dst = cv2.perspectiveTransform(pts, matrix)

                #corners of square
                cd1 = (dst[0][0][0], dst[0][0][1])
                cd2 = (dst[1][0][0], dst[1][0][1])
                cd3 = (dst[2][0][0], dst[2][0][1])
                cd4 = (dst[3][0][0], dst[3][0][1])

                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), cv2.LINE_AA)
                # cv2.circle(homography, cd1, 5, (0, 255, 0), -1)
                # cv2.circle(homography, cd2, 5, (0, 0, 255), -1)
                # cv2.circle(homography, cd3, 5, (0, 255, 255), -1)
                # cv2.circle(homography, cd4, 5, (255, 255, 0), -1)

                w_sum = 0
                h_sum = 0
                for i in range(4):
                    w_sum += dst[i][0][0]
                    h_sum += dst[i][0][1]
        
                centre = (int(w_sum/4), int(h_sum/4))
                cv2.circle(homography, centre, 5, (0, 255, 0), -1)
                cv2.imshow('frame', homography)
            else:
                cv2.imshow('frame', frame)

        else:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()