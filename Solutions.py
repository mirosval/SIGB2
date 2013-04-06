from __future__ import print_function
import cv2
import numpy as np
from pylab import *

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
    map = cv2.imread("Images/ITUMap.bmp")
    trace = np.loadtxt('GroundFloorData/trackingdata.dat')

    homography = [[  8.90361746e-01, 7.47992675e+00, -1.41155997e+02],
                  [ -1.59597293e+00, 3.02053067e+00, 3.18806955e+02],
                  [  1.65239983e-03, 1.57927362e-02, 1.00000000e+00]]
    sequence = cv2.VideoCapture("GroundFloorData/SunClipDS.avi")
    retval, image = sequence.read()

#    outputVideo = cv2.VideoWriter("Solutions/trace.avi", cv2.cv.FOURCC("i", "Y", "U", "V"), sequence.get(cv2.cv.CV_CAP_PROP_FPS), (map.shape[0], map.shape[1]))
    outputVideo = cv2.VideoWriter("Solutions/trace.avi", cv2.cv.FOURCC("X", "V", "I", "D"), sequence.get(cv2.cv.CV_CAP_PROP_FPS), (map.shape[0], map.shape[1]))

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

#        cv2.imshow("Trace", map)
#        cv2.waitKey(0)

        outputVideo.write(map)

        print(".", end="")
        if traceId % 24 == 0:
            print("%.2f%%" % round(traceId / totalFrames * 100, 2))

        retval, image = sequence.read()
        traceId += 1

#    cv2.waitKey(0)
