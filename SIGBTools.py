import cv2;
import cv2
import numpy as np
import pylab
from pylab import *
import matplotlib as mpl
import math
from scipy import linalg
''' This module contains sets of functions useful for basic image analysis and should be useful in the SIGB course.
Written and Assembled  (2012,2013) by  Dan Witzner Hansen, IT University.
'''

def getHomographyFromMouse(I1, I2, N=4):
    """
    getHomographyFromMouse(I1,I2,N=4)-> homography, mousePoints

    Calculates the homography from a plane in image I1 to a plane in image I2 by using the mouse to define corresponding points
    Returns: the 3x3 homography matrix and the set of corresponding points used to define the homography
    Parameters:    N>=4 is the number of expected mouse points in each image.
        when N<0: then the corners of image I1 will be used as input and thus only 4 mouse clicks are needed in I2

    Usage:   use left click to select a point and right click to remove the most recently selected point
    """
    # Copy images
    drawImg = []


    drawImg.append(copy(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)))
    drawImg.append(copy(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)))

    imagePoints = []
    firstImage = 0

    if(N < 0):
        N = 4  # Force 4 points to be selected
        firstImage = 1
        m, n, d = I1.shape
        # Define corner points
        imagePoints.append([(float(0.0), float(0.0)), (float(n), 0), (float(n), float(m)), (0, m)])

    if(math.fabs(N) < 4):
        print('at least 4 points are needed')

    # Make figure
    fig = figure(1)

    for k in range(firstImage, 2):
        ax = subplot(1, 2, k + 1)
        ax.imshow(drawImg[k])
        ax.axis('image')
        title("Click" + str(N) + " times in the  image")
        fig.canvas.draw()
        ax.hold('On')

        # Get mouse inputs
        imagePoints.append(fig.ginput(N, -1))

        # Draw selected points


        for p in imagePoints[k]:
           cv2.circle(drawImg[k], (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)
        ax.imshow(drawImg[k])
        for (x, y) in imagePoints[k]:
            plot(x, y, 'rx')
        fig.canvas.draw()

    # Convert to openCV format
    ip1 = np.array([[x, y] for (x, y) in imagePoints[0]])
    ip2 = np.array([[x, y] for (x, y) in imagePoints[1]])

    # Calculate homography
    H, mask = cv2.findHomography(ip1, ip2)
    return H, imagePoints

def getCircleSamples(center=(0, 0), radius=1, nPoints=30):
    ''' Samples a circle with center center = (x,y) , radius =1 and in nPoints on the circle.
    Returns an array of a tuple containing the points (x,y) on the circle and the curve gradient in the point (dx,dy)
    Notice the gradient (dx,dy) has unit length'''


    s = np.linspace(0, 2 * math.pi, nPoints)
    # points
    P = [(radius * np.cos(t) + center[0], np.sin(t) + center[1], np.cos(t), np.sin(t)) for t in s ]
    return P
def estimateHomography(points1, points2):
    """ Calculates a homography from one plane to another using
        a set of 4 points from each plane

        points1 is 4 points (nparray) on a plane
        points2 is the corresponding 4 points (nparray) on an other plane
    """
    # Check if there is sufficient points
    if len(points1) == 4 and len(points2) == 4:
        # Get x, y values
        x1, y1 = points1[0]
        x2, y2 = points1[1]
        x3, y3 = points1[2]
        x4, y4 = points1[3]

        # Get x tilde and y tilde values
        x_1, y_1 = points2[0]
        x_2, y_2 = points2[1]
        x_3, y_3 = points2[2]
        x_4, y_4 = points2[3]

        # Create matrix A
        A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1 * x_1, y1 * x_1, x_1],
            [0, 0, 0, -x1, -y1, -1, x1 * y_1, y1 * y_1, y_1],
            [-x2, -y2, -1, 0, 0, 0, x2 * x_2, y2 * x_2, x_2],
            [0, 0, 0, -x2, -y2, -1, x2 * y_2, y2 * y_2, y_2],
            [-x3, -y3, -1, 0, 0, 0, x3 * x_3, y3 * x_3, x_3],
            [0, 0, 0, -x3, -y3, -1, x3 * y_3, y3 * y_3, y_3],
            [-x4, -y4, -1, 0, 0, 0, x4 * x_4, y4 * x_4, x_4],
            [0, 0, 0, -x4, -y4, -1, x4 * y_4, y4 * y_4, y_4]])

        # Calculate SVD
        U, D, V = np.linalg.svd(A)

        # Get last row of V returned from SVD
        h = V [8]

        # Create homography matrix
        Homography = np.matrix([[h[0, 0], h[0, 1], h[0, 2]],
            [h[0, 3], h[0, 4], h[0, 5]],
            [h[0, 6], h[0, 7], h[0, 8]]])

        # Normalize homography
        Homography = Homography / Homography[2, 2]

        # Return matrix
        return Homography
def getImageSequence(fn, fastForward=2):
    '''Load the video sequence (fn) and proceeds, fastForward number of frames.'''
    cap = cv2.VideoCapture(fn)
    for t in range(fastForward):
        running, imgOrig = cap.read()  # Get the first frames
    return cap, imgOrig, running

def getLineCoordinates(p1, p2):
    "Get integer coordinates between p1 and p2 using Bresenhams algorithm"
    " When an image I is given the method also returns the values of I along the line from p1 to p2. p1 and p2 should be within the image I"
    " Usage: coordinates=getLineCoordinates((x1,y1),(x2,y2))"


    (x1, y1) = p1
    x1 = int(x1); y1 = int(y1)
    (x2, y2) = p2
    x2 = int(x2);y2 = int(y2)

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append([y, x])
        else:
            points.append([x, y])
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()

    retPoints = np.array(points)
    X = retPoints[:, 0];
    Y = retPoints[:, 1];


    return retPoints

class RegionProps:
    '''Class used for getting descriptors of contour-based connected components 
        
        The main method to use is: CalcContourProperties(contour,properties=[]):
        contour: a contours found through cv2.findContours
        properties: list of strings specifying which properties should be calculated and returned
        
        The following properties can be specified:
        
        Area: Area within the contour  - float 
        Boundingbox: Bounding box around contour - 4 tuple (topleft.x,topleft.y,width,height) 
        Length: Length of the contour
        Centroid: The center of contour: (x,y)
        Moments: Dictionary of moments: see 
        Perimiter: Permiter of the contour - equivalent to the length
        Equivdiameter: sqrt(4*Area/pi)
        Extend: Ratio of the area and the area of the bounding box. Expresses how spread out the contour is
        Convexhull: Calculates the convex hull of the contour points
        IsConvex: boolean value specifying if the set of contour points is convex
        
        Returns: Dictionary with key equal to the property name
        
        Example: 
             contours, hierarchy = cv2.findContours(I, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
             goodContours = []
             for cnt in contours:
                vals = props.CalcContourProperties(cnt,['Area','Length','Centroid','Extend','ConvexHull'])
                if vals['Area']>100 and vals['Area']<200
                 goodContours.append(cnt)
        '''
    def __calcArea(self, m, c):
        return cv2.contourArea(c)  # ,m['m00']
    def __calcLength(self, c):
        return cv2.arcLength(c, True)
    def __calcPerimiter(self, c):
         return cv2.arcLength(c, True)
    def __calcBoundingBox(self, c):
        return cv2.boundingRect(c)
    def __calcCentroid(self, m):
        if(m['m00'] != 0):
            retVal = (m['m10'] / m['m00'], m['m01'] / m['m00'])
        else:
            retVal = (-1, -1)
        return retVal

    def __calcEquivDiameter(self, contur):
        Area = self.__calcArea(m)
        return np.sqrt(4 * Area / np.pi)
    def __calcExtend(self, m, c):
        Area = self.__calcArea(m, c)
        BoundingBox = self.__calcBoundingBox(c)
        return Area / (BoundingBox[2] * BoundingBox[3])
    def __calcConvexHull(self, m, c):
         # try:
             CH = cv2.convexHull(c)
             # ConvexArea  = cv2.contourArea(CH)
             # Area =  self.__calcArea(m,c)
             # Solidity = Area/ConvexArea
             return {'ConvexHull':CH}  # {'ConvexHull':CH,'ConvexArea':ConvexArea,'Solidity':Solidity}
         # except:
         #    print "stuff:", type(m), type(c)

    def CalcContourProperties(self, contour, properties=[]):
        failInInput = False;
        propertyList = []
        contourProps = {};
        for prop in properties:
            prop = str(prop).lower()
            m = cv2.moments(contour)  # Always call moments
            if (prop == 'area'):
                contourProps.update({'Area':self.__calcArea(m, contour)});
            elif (prop == "boundingbox"):
                contourProps.update({'BoundingBox':self.__calcBoundingBox(contour)});
            elif (prop == "length"):
                contourProps.update({'Length':self.__calcLength(contour)});
            elif (prop == "centroid"):
                contourProps.update({'Centroid':self.__calcCentroid(m)});
            elif (prop == "moments"):
                contourProps.update({'Moments':m});
            elif (prop == "perimiter"):
                contourProps.update({'Perimiter':self.__calcPermiter(contour)});
            elif (prop == "equivdiameter"):
                contourProps.update({'EquivDiameter':self.__calcEquiDiameter(m, contour)});
            elif (prop == "extend"):
                contourProps.update({'Extend':self.__calcExtend(m, contour)});
            elif (prop == "convexhull"):  # Returns the dictionary
                contourProps.update(self.__calcConvexHull(m, contour));
            elif (prop == "isConvex"):
                    contourProps.update({'IsConvex': cv2.isContourConvex(contour)});
            elif failInInput:
                    pass
            else:
                print "--"*20
                print "*** PROPERTY ERROR " + prop + " DOES NOT EXIST ***"
                print "THIS ERROR MESSAGE WILL ONLY BE PRINTED ONCE"
                print "--"*20
                failInInput = True;
        return contourProps


class ROISelector:

    def __resetPoints(self):
        self.seed_Left_pt = None
        self.seed_Right_pt = None

    def __init__(self, inputImg):
        self.img = inputImg.copy()
        self.seed_Left_pt = None
        self.seed_Right_pt = None
        self.winName = 'SELECT AN AREA'
        self.help_message = '''This function returns the corners of the selected area as: [(UpperLeftcorner),(LowerRightCorner)]
        Use the Right Button to set Upper left hand corner and and the Left Button to set the lower righthand corner.
        Click on the image to set the area
        Keys:
          Enter/SPACE - OK
          ESC   - exit (Cancel)
        '''

    def update(self):
        if (self.seed_Left_pt is None) | (self.seed_Right_pt is None):
            cv2.imshow(self.winName, self.img)
            return

        flooded = self.img.copy()
        cv2.rectangle(flooded, self.seed_Left_pt, self.seed_Right_pt, (0, 0, 255), 1)
        cv2.imshow(self.winName, flooded)



    def onmouse(self, event, x, y, flags, param):

        if  flags & cv2.EVENT_FLAG_LBUTTON:
            self.seed_Left_pt = x, y
    #        print seed_Left_pt

        if  flags & cv2.EVENT_FLAG_RBUTTON:
            self.seed_Right_pt = x, y
    #        print seed_Right_pt

        self.update()
    def setCorners(self):
        points = []

        UpLeft = (min(self.seed_Left_pt[0], self.seed_Right_pt[0]), min(self.seed_Left_pt[1], self.seed_Right_pt[1]))
        DownRight = (max(self.seed_Left_pt[0], self.seed_Right_pt[0]), max(self.seed_Left_pt[1], self.seed_Right_pt[1]))
        points.append(UpLeft)
        points.append(DownRight)
        return points

    def SelectArea(self, winName='SELECT AN AREA', winPos=(400, 400)):  # This function returns the corners of the selected area as: [(UpLeftcorner),(DownRightCorner)]
        self.__resetPoints()
        self.winName = winName
        print  self.help_message
        self.update()
        cv2.namedWindow(self.winName, cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_AUTOSIZE
        cv2.setMouseCallback(self.winName, self.onmouse)
        cv2.cv.MoveWindow(self.winName, winPos[0], winPos[1])
        while True:
            ch = cv2.waitKey()

            if ch == 27:  # Escape
                cv2.destroyWindow(self.winName)
                return None, False
                break
            if ((ch == 13) or (ch == 32)):  # enter or space key
                cv2.destroyWindow(self.winName)
                return self.setCorners(), True
                break


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def anorm2(a):
    return (a * a).sum(-1)

def anorm(a):
    return np.sqrt(anorm2(a))

def to_rect(a):
    a = np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return np.array(a, np.float64).reshape(2, 2)

def rect2rect_mtx(src, dst):
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = np.float64([[ cx, 0, tx],
                    [  0, cy, ty],
                    [  0, 0, 1]])
    return M

def rotateImage(I, angle):
    "Rotate the image, I, angle degrees around the image center"
    size = I.shape
    image_center = tuple(np.array(size) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center[0:2], angle, 1)
    result = cv2.warpAffine(image, rot_mat, dsize=size[0:2], flags=cv2.INTER_LINEAR)
    return result

def lookat(eye, target, up=(0, 0, 1)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= anorm(fwd)
    right = np.cross(fwd, up)
    right /= anorm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def mtx2rvec(R):
    w, u, vt = cv2.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:, 0] * w[0]  # same as np.dot(R, vt[0])
    c = np.dot(vt[0], p)
    s = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(s, c)

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, linetype=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), linetype=cv2.CV_AA)

class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()
        else:
            self.prev_pt = None


# palette data from matplotlib/_cm.py
_jet_data = {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1),
                     (1, 0.5, 0.5)),
           'green': ((0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1),
                     (0.91, 0, 0), (1, 0, 0)),
           'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0),
                     (1, 0, 0))}

cmap_data = { 'jet' : _jet_data }

def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x + eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T * 255)

def nothing(*arg, **kw):
    pass

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def imWeight(x, c):
    return  1 - (np.abs(x - c) / float(c))

def HatFunction(I):
    imSize = I.shape; m = imSize[0]; n = imSize[1];
    weightMatrix = np.array([ [(imWeight(i, n / 2) * imWeight(j, m / 2)) for i in range(n)] for j in range(m)])
    return weightMatrix


def toHomogenious(points):
    """ Convert a set of points (dim*n array) to
        homogeneous coordinates. """
    return vstack((points, ones((1, points.shape[1]))))

def normalizeHomogenious(points):
    """ Normalize a collection of points in
    homogeneous coordinates so that last row = 1. """
    for row in points:
        row /= points[-1]
    return points
def H_from_points(fp, tp):
    """ Find homography H, such that fp is mapped to tp
    using the linear DLT method. Points are conditioned automatically. """
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    # condition points (important for numerical reasons)

    # --from points
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    T1 = diag([1 / maxstd, 1 / maxstd, 1])
    T1[0][2] = -m[0] / maxstd
    T1[1][2] = -m[1] / maxstd
    fp = dot(T1, fp)

    # --to points--
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    T2 = diag([1 / maxstd, 1 / maxstd, 1])
    T2[0][2] = -m[0] / maxstd
    T2[1][2] = -m[1] / maxstd
    tp = dot(T2, tp)
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i], tp[0][i] * fp            [1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i], tp[1][i] * fp              [1][i], tp[1][i]]

    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))
    # decondition
    H = dot(linalg.inv(T2), dot(H, T1))  # normalize and return
    return H / H[2, 2]


def calibrateCamera(camNum=0, nPoints=5, patternSize=(9, 6), saveImage=False):
    ''' CalibrateCamera captures images from camera (camNum)
        The user should press spacebar when the calibration pattern
        is in view.
        When saveImage is a boolean it indicates whether the images used for calibration should be saved
        When it is True the image will be save into a default filename. When saveImage is a string the images
        will be saved using the string as the filename.
    '''
    print('click on the image window and then press the space key to take samples')
    cv2.namedWindow("camera", 1)
    pattern_size = patternSize
    n = nPoints  # number of images before calibration
    # temp=n
    calibrated = False
    square_size = 1
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    camera_matrix = np.zeros((3, 3))
    dist_coefs = np.zeros(4)
    rvecs = np.zeros((3, 3))
    tvecs = np.zeros((3, 3))

    obj_points = []
    img_points = []

    capture = cv2.VideoCapture(camNum)
    imgCnt = 0
    running = True
    while running:

        ret, imgOrig = capture.read()
        img = imgOrig.copy()
        h, w = img.shape[:2]
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if (calibrated == False):
            found, corners = cv2.findChessboardCorners(imgGray, pattern_size)
        ch = cv2.waitKey(1)

        if(ch == 27):  # ESC
            running = False
            found = False
            calibrated = False
            return (calibrated, None, None, None)

        if (found != 0) & (n > 0):
            cv2.drawChessboardCorners(img, pattern_size, corners, found)
            if ((ch == 13) or (ch == 32)):  # enter or space key :
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(imgGray, corners, (5, 5), (-1, -1), term)

                img_points.append(corners.reshape(-1, 2))
                obj_points.append(pattern_points)
                n = n - 1
                imgCnt = imgCnt + 1;
                print('sample %s taken') % (imgCnt)
                if(saveImage != False):
                    if(saveImage == True):
                        fileName = 'CalibrationImage' + str(imgCnt) + ".jpg"
                    else:
                        fileName = saveImage + str(imgCnt) + ".jpg"
                    print("saving Image " + fileName)
                    cv2.imwrite(fileName, imgOrig)
                if n == 0:
    #                print( img_points)
    #                print(obj_points)
                    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), camera_matrix, dist_coefs, flags=0)
    #               print "RMS:", rms
                    print "camera matrix:\n", camera_matrix
                    print "distortion coefficients: ", dist_coefs
                    calibrated = True

                    return (calibrated, camera_matrix, dist_coefs, rms)

        elif(found == 0) & (n > 0):
            print("chessboard not found")


        # if (calibrated):
        #    img=cv2.undistort(img, camera_matrix, dist_coefs )
        #    found,corners=cv2.findChessboardCorners(imgGray, pattern_size  )
        #    if (found!=0):
        #        cv2.drawChessboardCorners(img, pattern_size, corners,found)

        cv2.imshow("camera", img)


class Camera:
    """ Class for representing pin-hole cameras. """

    def __init__(self, P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None  # calibration matrix
        self.R = None  # rotation
        self.t = None  # translation
        self.c = None  # camera center


    def project(self, X):
        """    Project points in X (4*n array) and normalize coordinates. """

        x = dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x


    def factor(self):
        """    Factorize the camera matrix into K,R,t as P = K[R|t]. """

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])

        # make diagonal of K positive
        T = diag(sign(diag(K)))

        self.K = dot(K, T)
        self.R = dot(T, R)  # T is its own inverse
        self.t = dot(linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t


    def center(self):
        """    Compute and return the camera center. """

        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -dot(self.R.T, self.t)
            return self.c


    def calibrate_from_points(x1, x2):

        return self.K


    def simple_calibrate(a, b):

        return self.K


# helper functions

def rotation_matrix(a):
    """    Creates a 3D rotation matrix for rotation
         around the axis of the vector a. """
    R = eye(4)
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return R


def rq(A):
    from scipy.linalg import qr

    Q, R = qr(flipud(A).T)
    R = flipud(R.T)
    Q = Q.T

    return R[:, ::-1], Q[::-1, :]
