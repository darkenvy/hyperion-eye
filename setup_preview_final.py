import sys
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from random import randint
from PIL import Image
import base64
import json
import socket

# ------------------------------- Configuration ------------------------------- #

# Replace the next 3 assignments to the results retrieved from fisheye_calibrate.py
DIM=(2592, 1944)
K=np.array([[1259.041938650184, 0.0, 1350.8011212450504],
            [0.0, 1263.2227906651265, 896.9567943445292],
            [0.0, 0.0, 1.0]])
D=np.array([[-0.02553723406094832], [0.02138072637573416], [-0.11357946018325828], [0.10334136402735668]])

TV_ASPECT_RATIO = 1.75

# Find these points by running setup_find_perspective_points.py and using the mounse
# to determine coordinates of the 4 corners of the tv
VECTOR_A = [217,48] # Top-Left of Screen
VECTOR_B = [443,40] # Top-Right of Screen
VECTOR_C = [115,122] # Bottom-Left of Screen
VECTOR_D = [572,116] # Bottom-Right of Screen

# Refer to https://picamera.readthedocs.io/en/release-1.12/fov.html and set your
# resolution to the highest FoV (field of fiew). Pixels dont matter; its the FoV.
# The final image sent to hyperion is less than 100x80 anyways.
# The provided numers are for the v1 pi camera
RESOLUTION = (1296, 972) # Set to your 
FRAME_RATE = 20 # default: 20. Decrease if having latency issues

# -------------------------------- Source Code -------------------------------- #

def unfisheye(cv2_img, balance=0.0, dim2=None, dim3=None):
    img = cv2_img
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return undistorted_img

def resize(img, resize_factor):
  NEW_WIDTH = int(camera.resolution[0] * resize_factor)
  NEW_HEIGHT = int(camera.resolution[1] * resize_factor)
  return cv2.resize(img, (NEW_WIDTH, NEW_HEIGHT), interpolation = cv2.INTER_AREA)

def perspective_transform(img, from_coords):
  pts1 = np.float32(from_coords)
  pts2 = np.float32([[0,         0], [TV_WIDTH,         0],
                     [0, TV_HEIGHT], [TV_WIDTH, TV_HEIGHT]])
  matrix = cv2.getPerspectiveTransform(pts1, pts2)

  # execute transformation on matrix
  return cv2.warpPerspective(img, matrix, (TV_WIDTH, TV_HEIGHT))

def extend_fill_border(img, horizontal_multiplier, vertical_multiplier):
  BLACK = [0,0,0]

  top = int(vertical_multiplier * img.shape[0])
  left = int(horizontal_multiplier * img.shape[1])
  bottom = top
  right = left
  
  return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT, None, BLACK)

def draw_vector_points_for_reference(img):
  cv2.circle(img, (VECTOR_A[0], VECTOR_A[1]), 5, (0, 0, 255), -1)
  cv2.circle(img, (VECTOR_B[0], VECTOR_B[1]), 5, (0, 0, 255), -1)
  cv2.circle(img, (VECTOR_C[0], VECTOR_C[1]), 5, (0, 0, 255), -1)
  cv2.circle(img, (VECTOR_D[0], VECTOR_D[1]), 5, (0, 0, 255), -1)

def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

def main():
  # capture frames from the camera
  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array # grab the raw NumPy array representing the image, then initialize the timestamp

    # --------------------------------- Resize --------------------------------- #
    img = resize(img, 0.25)

    # ------------------------------- De-Fisheye -------------------------------- #
    img = unfisheye(img, 1)

    # ------------------------------ Border Extend ------------------------------ #
    img = extend_fill_border(img, 0.5, 0)

    # -------------------------- Perspective Transform -------------------------- #
    img = perspective_transform(img, [VECTOR_A, VECTOR_B, VECTOR_C, VECTOR_D])

    # ------------------------------- Show Image ------------------------------- #
    cv2.imshow("Original", img) # show the frame
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0) # clear the stream in preparation for the next frame

    if key == ord("q"): # if the `q` key was pressed, break from the loop
      break

  s.close()

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'

# initialize the camera and grab a reference to the raw camera capture
HOST = socket.gethostname()
PORT = 19444
SENSOR_MODE = 4
TV_WIDTH = 500
TV_HEIGHT = int(TV_WIDTH / TV_ASPECT_RATIO)
TCP_BASE_IMG_CMD = {
  "command":"image",
  "imagewidth": TV_WIDTH,
  "imageheight": TV_HEIGHT,
  "priority":500,
}

camera = PiCamera(0, 'none', False, RESOLUTION, FRAME_RATE, SENSOR_MODE)
camera.exposure_mode = 'night'

# Connect to hyperiond JSON server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

rawCapture = PiRGBArray(camera, size=camera.resolution)
time.sleep(0.1) # allow the camera to warmup
main()
