import cv2
import numpy as np
import sys

# You should replace these 3 lines with the output in calibration step
DIM=(2592, 1944)
K=np.array([[1259.041938650184, 0.0, 1350.8011212450504], [0.0, 1263.2227906651265, 896.9567943445292], [0.0, 0.0, 1.0]])
D=np.array([[-0.02553723406094832], [0.02138072637573416], [-0.11357946018325828], [0.10334136402735668]])

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)