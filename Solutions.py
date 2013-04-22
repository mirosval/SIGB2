from __future__ import print_function
import cv2
import numpy as np
from pylab import *
from SIGBTools import getImageSequence
from SIGBTools import Camera
from cubePoints import cube_points
import SIGBTools


def drawTrace(homography, sourceTrace, destinationImage):
    '''
    Transform points from source trace into the destinationImage using homography
    
    Params:
        homography: 3x3 matrix
        sourceTrace: list of N 12-element person descriptors
        destinationImage: numpy array image to draw to
    Returns:
        numpy array image with the trace drawn into it
    '''

    # iterate over all positions in the trace
    for frameTrackingData in sourceTrace:
        # indices 4,5 are x and y of top left point of the leg rectangle
        # indices 6,7 are x and y of bottom right of that triangle
        # we need the average of the x values of the two, and the y value of
        # the latter one to get a point that lies in the middle bottom of that
        # rectangle
        point = ((frameTrackingData[4] + frameTrackingData[6]) / 2.0, frameTrackingData[7], 1)

        # dot product of the homography and our calculated point will
        # yield the corresponding point in the destinationImage
        destPoint = dot(homography, point)

        # denormalize point
        w = destPoint[2]
        destPoint = (int(destPoint[0] / w), int(destPoint[1] / w))

        # draw it into the image
        cv2.circle(destinationImage, destPoint, 1, (0, 0, 255), -1)

    return destinationImage

def displayTraceImage():
    '''
    Mandatory Assignment #2 pg. 2 Question 9.
    Make a function DisplayTrace that uses the estimated homography to display the 
    trace of the person in the overview map. Notice, you should only map points that 
    are on the ground floor to the overview map (e.g. the feet and not the head).
    The example function showImageandPlot(N) should give you sufficient information 
    on how you can display and save the image data.
    
    Solution:
    Using Tracking data from GroundFloorData/trackingdata.dat, homography
    obtained from Assignment2.py simpleTextureMap() and map from Images/ITUMap.bmp
    draw trace using the drawTrace function defined above draw the trace to the map
    and write out the resulting map with trace and original camera image overlaid
    '''

    # Load everything
    trace = np.loadtxt('GroundFloorData/trackingdata.dat')
    homography = [[  8.90361746e-01, 7.47992675e+00, -1.41155997e+02],
                  [ -1.59597293e+00, 3.02053067e+00, 3.18806955e+02],
                  [  1.65239983e-03, 1.57927362e-02, 1.00000000e+00]]

    homography = np.array(homography)

    map = cv2.imread("Images/ITUMap.bmp")

    # Draw trace onto the map image
    map = drawTrace(homography, trace, map)

    # Draw original sequence image as an overlay
    sequence = cv2.VideoCapture("GroundFloorData/SunClipDS.avi")
    retval, image = sequence.read()

    h, w, d = map.shape
    overlay = cv2.warpPerspective(image, homography, (w, h))
    map = cv2.addWeighted(map, 0.5, overlay, 0.5, 0)

    # write out to solutions folder
    cv2.imwrite("Solutions/trace_map.png", map)
    np.save("Solutions/trace_homography", homography)

    # display on screen as well
    cv2.imshow("Trace", map)
    cv2.waitKey(0)

def traceVideo():
    '''
    Trace the person in the video and draw an overlay map
    '''
    map = cv2.imread("Images/ITUMap.bmp")
    trace = np.loadtxt('GroundFloorData/trackingdata.dat')

    # homography from the video image to the overview map (pre-calculated during calibration)
    homography = [[  8.90361746e-01, 7.47992675e+00, -1.41155997e+02],
                  [ -1.59597293e+00, 3.02053067e+00, 3.18806955e+02],
                  [  1.65239983e-03, 1.57927362e-02, 1.00000000e+00]]

    sequence = cv2.VideoCapture("GroundFloorData/SunClipDS.avi")
    retval, image = sequence.read()

    outputVideo = cv2.VideoWriter("Solutions/trace.avi", cv2.cv.FOURCC("X", "V", "I", "D"), sequence.get(cv2.cv.CV_CAP_PROP_FPS), (map.shape[1], map.shape[0]))

    tx = map.shape[1] - image.shape[1]
    ty = map.shape[0] - image.shape[0]

    totalFrames = sequence.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    print("Processing Video")

    traceId = 0
    while retval:
        for y in range(ty, map.shape[0]):
            for x in range(tx, map.shape[1]):
                map[y][x] = image[y - ty][x - tx]

        frameTrackingData = trace[traceId]

        point = ((frameTrackingData[4] + frameTrackingData[6]) / 2.0, frameTrackingData[7], 1)
        destPoint = dot(homography, point)

        # denormalize point
        w = destPoint[2]
        destPoint = (int(destPoint[0] / w), int(destPoint[1] / w))

        # draw it into the image
        cv2.circle(map, destPoint, 2, (0, 0, 255), -1)

        outputVideo.write(map)

        print(".", end="")
        if traceId % 24 == 0:
            print("%.2f%%" % round(traceId / totalFrames * 100, 2))

        retval, image = sequence.read()
        traceId += 1

    outputVideo.release()

def texturemapGroundFloor(SequenceInputFile):
    '''
    get four points in the map and overview a logo on the sequence
    '''
    sequence, I2, retval = getImageSequence(SequenceInputFile)
    I1 = cv2.imread('Images/ITULogo.jpg')
    H, Points = SIGBTools.getHomographyFromMouse(I1, I2, -4) #get 4 points from mouse input 
    h, w, d = I2.shape
    if(retval):
        cv2.imshow("Overlayed Image", I2)
    print("SPACE: Run/Pause")
    print("Q or ESC: Stop")
    running = True
    while(retval):
        ch = cv2.waitKey(1)
        # video controls
        if(ch == 32):  # Spacebar
            if(running):
                running = False
            else:
                running = True
        if ch == 27:
            break
        if(ch == ord('q')):
            break
        if(running):
            retval, I2 = sequence.read()
            if(retval):  # if there is an image
                overlay = cv2.warpPerspective(I1, H, (w, h)) #get the perspective image for overlaying on the video
                M = cv2.addWeighted(I2, 0.5, overlay, 0.5, 0) #overlay the video with the image
                cv2.imshow("Overlayed Image", M) #show the result

def texturemapGrid():
    """ Skeleton for texturemapping on a video sequence"""
    fn = 'GridVideos/grid1.mp4'
    cap = cv2.VideoCapture(fn)
    texture = cv2.imread('Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)
    running, imgOrig = cap.read()
    cv2.imshow("win2", imgOrig)
    pattern_size = (9, 6)
    idx = [0, 8, 45, 53]
    while(running):
    # load Tracking data
        running, imgOrig = cap.read()
        if(running):
            imgOrig = cv2.pyrDown(imgOrig)
            gray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            if found:
                #term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                #cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                H, Points = getHomography(texture, corners) #find the homography matrix to overview the texture on patterns corners
                overlay = cv2.warpPerspective(texture, H, (imgOrig.shape[1], imgOrig.shape[0])) #get the perspective image for overlaying on the video
                M = cv2.addWeighted(imgOrig, 0.7, overlay, 0.8, 0) #overlay the video with the image
            cv2.imshow("win2", M)
            cv2.waitKey(1)

def getHomography(I1, corners1):
    """ get the homography matrix for an image by using four points"""
    imagePoints = []
    m, n, d = I1.shape
    imagePoints.append([(float(0.0), float(0.0)), (float(n), 0), (float(n), float(m)), (0, m)]) #append image corners point to an array
    imagePoints.append([ (float(corners1[0, 0, 0]), float(corners1[0, 0, 1])), (float(corners1[8, 0, 0]), float(corners1[8, 0, 1])), (float(corners1[53, 0, 0]), float(corners1[53, 0, 1])), (float(corners1[45, 0, 0]), float(corners1[45, 0, 1]))]) #append patterns corners point to the previous array
    ip1 = np.array([[x, y] for (x, y) in imagePoints[0]]) #select and convert part of the array to a numpy array
    ip2 = np.array([[x, y] for (x, y) in imagePoints[1]]) #select and convert part of the array to a numpy array
    H, mask = cv2.findHomography(ip1, ip2) #get the homography
    return H, imagePoints

def cameraCalibration():
    camNum = 0  # The number of the camera to calibrate
    nPoints = 7  # number of images used for the calibration (space presses)
    patternSize = (9, 6)  # size of the calibration pattern
    saveImage = "Solutions/cam_calibration"

    calibrated, camera_matrix, dist_coefs, rms = SIGBTools.calibrateCamera(camNum, nPoints, patternSize, saveImage)
    K = camera_matrix

    saveCameraCalibration(camera_matrix, dist_coefs)

    cam1 = Camera(np.hstack((K, np.dot(K, np.array([[0], [0], [-1]])))))
    cam1.factor()
    # Factor projection matrix into intrinsic and extrinsic parameters
    print("K=" + cam1.K)
    print("R=" + cam1.R)
    print("t" + cam1.t)

    if (calibrated):
        capture = cv2.VideoCapture(camNum)
        running = True
        while running:
            running, img = capture.read()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ch = cv2.waitKey(1)
            if(ch == 27) or (ch == ord('q')):  # ESC
                running = False
            img = cv2.undistort(img, camera_matrix, dist_coefs)
            found, corners = cv2.findChessboardCorners(imgGray, patternSize)
            if (found != 0):
                cv2.drawChessboardCorners(img, patternSize, corners, found)
            cv2.imshow("Calibrated", img)

def saveCameraCalibration(cameraCalibration, cameraDistortionCoefficients):
#    cameraCalibration = [[ 639.90749935, 0., 316.47044387],
#                          [   0., 641.2789932, 242.42122905],
#                          [   0., 0., 1.        ]]
#
#    cameraDistortionCoefficients = [[ -4.44612659e-02],
#                                     [  8.77982500e-01],
#                                     [ -2.53953866e-03],
#                                     [  1.35770339e-03],
#                                     [ -3.06879241e+00]]

    np.save("Solutions/PMatrix", cameraCalibration)
    np.save("Solutions/distCoef", cameraDistortionCoefficients)

def loadCameraCalibration():
    camera_matrix = np.load("Solutions/PMatrix.npy")
    coef = np.load("Solutions/distCoef.npy")

    return camera_matrix, coef

def augmentImages():
    '''
    We augment arbitrary images with the chessboard with a cube
    using a two-camera approach
    '''
    # load calibration
    cam_calibration, distCoef = loadCameraCalibration()

    # choose points on the chessboard pattern
    idx = np.array([1, 7, 37, 43])

    # load calibration pattern and transform the image
    calibration_pattern = cv2.imread("Images/CalibrationPattern.png")
    calibration_pattern = cv2.resize(calibration_pattern, (640, 480))
    calibration_pattern = cv2.cvtColor(calibration_pattern, cv2.COLOR_BGR2GRAY)

    # get corners from the calibration pattern
    found, calibrationCorners = cv2.findChessboardCorners(calibration_pattern, (9, 6))

    # load images to be augmented
    images = []
    for i in range(1, 8):
        images.append(cv2.imread("Solutions/cam_calibration{}.jpg".format(i)))

    # augment the images one by one
    for image_id, image in enumerate(images):

        # find the same corners as we had found previously in the
        # chessboard pattern itself, only this one is in the video
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (9, 6))

        if not found:
            continue

        # load up coords in the respective images
        imagePoints = []
        calibrationPoints = []
        for i in idx:
            p = corners[i][0]
            cp = calibrationCorners[i][0]
            imagePoints.append(p)
            calibrationPoints.append(cp)

        imagePoints = np.array(imagePoints)
        calibrationPoints = np.array(calibrationPoints)

#         cv2.imshow('calibration image', calibration_pattern)

        # Create 1st camera, this one is looking at the pattern image
        cam1 = Camera(hstack((cam_calibration, dot(cam_calibration, np.array([[0], [0], [-1]])))))
        cam1.factor()

        # Create the cube
        cube = cube_points([0, 0, 0.1], 0.3)

        # Project the bottom square of the cube, this will transform
        # point coordinates from the object space to the calibration
        # world space where the camera looks
        calibration_rect = cam1.project(SIGBTools.toHomogenious(cube[:, :5]))

        # Calculate the homography from the corners in the calibration image
        # to the same points in the image that we want to project the cube to
        homography = SIGBTools.estimateHomography(calibrationPoints, imagePoints)

        # Transform the rect from the calibration image world space to the final
        # image world space
        transRect = SIGBTools.normalizeHomogenious(dot(homography, calibration_rect))

        # Create the second camera, looking into the world of the final image
        cam2 = Camera(dot(homography, cam1.P))

        # Recalculate the projection matrix
        calibrationInverse = np.linalg.inv(cam_calibration)
        rot = dot(calibrationInverse, cam2.P[:, :3])

        # reassemble the rotation translation matrix
        r1, r2, t = tuple(np.hsplit(rot, 3))
        r3 = cross(r1.T, r2.T).T
        rotationTranslationMatrix = np.hstack((r1, r2, r3, t))

        # Create the projection matrix
        cam2.P = dot(cam_calibration, rotationTranslationMatrix)
        cam2.factor()

        # project the cube using the 2nd camera
        cube = cube_points([0, 0, 0.1], 0.3)
        box = cam2.project(SIGBTools.toHomogenious(cube))

        for i in range(1, 17):
            x1 = box[0, i - 1]
            y1 = box[1, i - 1]
            x2 = box[0, i]
            y2 = box[1, i]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # save image
        cv2.imwrite("Solutions/augmentation{}.png".format(image_id), image)

        cv2.imshow("Test", image)
        cv2.waitKey(0)


def texturemapGridSequence():
    """ Skeleton for texturemapping on a video sequence"""
    fn = 'GridVideos/grid1.mp4'
    cap = cv2.VideoCapture(fn)
    drawContours = True;

    texture = cv2.imread('Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)


    mTex, nTex, t = texture.shape

    texturePoints = np.array([[0, 0],
                              [0, nTex],
                              [mTex, 0],
                              [mTex, nTex]], dtype=np.float32)

    # load Tracking data
    running, imgOrig = cap.read()
    mI, nI, t = imgOrig.shape

    cv2.imshow("win2", imgOrig)

    pattern_size = (9, 6)

    idx = [0, 8, 45, 53]
    while(running):
    # load Tracking data
        running, imgOrig = cap.read()
        if(running):
            imgOrig = cv2.pyrDown(imgOrig)
            gray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
#                 cv2.drawChessboardCorners(imgOrig, pattern_size, corners, found)

                imagePoints = np.array([[corners[0][0][0], corners[0][0][1]],
                                        [corners[8][0][0], corners[8][0][1]],
                                        [corners[45][0][0], corners[45][0][1]],
                                        [corners[53][0][0], corners[53][0][1]]], dtype=np.float32)

                homography = cv2.getPerspectiveTransform(texturePoints, imagePoints)

                overlay = cv2.warpPerspective(texture, homography, (imgOrig.shape[1], imgOrig.shape[0]))

                imgOrig = cv2.addWeighted(imgOrig, 0.5, overlay, 0.5, 0)

#                 for y, row in enumerate(imgOrig):
#                     for x, color in enumerate(row):
#                         if overlay[y][x][0] != 0 and overlay[y][x][1] != 0 and overlay[y][x][2] != 0:
#                              imgOrig[y][x] = overlay[y][x]

                for t in idx:
                    cv2.circle(imgOrig, (int(corners[t, 0, 0]), int(corners[t, 0, 1])), 10, (255, t, t))
            cv2.imshow("win2", imgOrig)
            cv2.waitKey(1)


def realisticTexturemapSol(scale):
    hgm = np.load("Solutions/trace_homography.npy")

    cap = cv2.VideoCapture("GroundFloorData/SunClipDS.avi")
    retval, image = cap.read()

    map = cv2.imread("Images/ITUMap.bmp")
    tex = cv2.imread("Images/ITULogo.jpg")

    tex = cv2.resize(tex, (int(tex.shape[1] * scale), int(tex.shape[0] * scale)))

    cv2.imshow("Map", map)

    def onMouse(event, x, y, flags, output):
        if event == 1:

            # point in the map image where the user clicked
            point = (x, y)

            # Homography from ground -> map inverted is
            # homography from map -> ground
            hmg = np.linalg.inv(hgm)

            # source points are points in the overview map
            width = tex.shape[1]
            height = tex.shape[0]
            source = [[x - width / 2, y - height / 2],
                      [x + width / 2, y - height / 2],
                      [x - width / 2, y + height / 2],
                      [x + width / 2, y + height / 2]]

            source = np.array(source, dtype=float32)

            # dest points are points in the video image
            dest = []
            for point in source:
                p = np.append(point, [1]).T
                p = dot(hmg, p).T

                p[0] = p[0] / p[2]
                p[1] = p[1] / p[2]

                dest.append(p[:2])

            dest = np.array(dest, dtype=float32)

            # tex source points are corners of the texture image
            texSource = [[0, 0],
                         [tex.shape[1], 0],
                         [0, tex.shape[0]],
                         [tex.shape[1], tex.shape[0]]]
            texSource = np.array(texSource, dtype=float32)

            # find the homography from the texture image to the video image
            transform, mask = cv2.findHomography(texSource, dest)

            # draw the image into perspective
            overlay = cv2.warpPerspective(tex, transform, (image.shape[1], image.shape[0]))

            # combine the results
            result = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

            # draw the result
            newMap = np.copy(map)
            for p in source:
                cv2.circle(newMap, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

            for p in dest:
                cv2.circle(result, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

            cv2.imshow("Map", newMap)
            cv2.imshow("Result", result)

    cv2.setMouseCallback("Map", onMouse)

    cv2.waitKey(0)
