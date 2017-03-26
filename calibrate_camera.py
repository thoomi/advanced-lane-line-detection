"""Collection of functions to determine a cameras distorion matrix"""
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Adjustable constans and variables
# =============================================================================
calibration_images = glob.glob('./camera_cal/calibration*.jpg')
chessboard_shape = (9, 6)


# =============================================================================
# Process each calibration image and save found chessboard points
# =============================================================================
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

for filename in calibration_images:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

    if ret is True:
        objpoints.append(objp)
        imgpoints.append(corners)


# =============================================================================
# Calculate, test and save distortion coefficients and matrix
# =============================================================================
img = cv2.imread('test_images/test1.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("camera_cal/calibration_coefficients.p", "wb"))

# Plot an example image to verify the calibration visually
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img[..., ::-1])
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst[..., ::-1])
ax2.set_title('Undistorted Image', fontsize=30)

plt.show()
