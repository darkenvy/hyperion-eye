  # Read an example image and acquire its size
  # img = cv2.imread("calibration_samples/2016-07-13-124020.jpg")
  # h, w = img.shape[:2]

  # # Generate new camera matrix from parameters
  # newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)

  # # Generate look-up tables for remapping the camera image
  # mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)

  # # Remap the original image to a new image
  # newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)



  # h,w = img.shape[:2]
  # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, d, numpy.eye(3), K, DIM, cv2.CV_16SC2)
  # newimg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
