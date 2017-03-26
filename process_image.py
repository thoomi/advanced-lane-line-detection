"""Process images based on the defined pipeline"""
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Load undistort matrix, define warp points and load test image
# =============================================================================
dist_pickle = pickle.load(open("./camera_cal/calibration_coefficients.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

src_top_left = [600, 450]
src_top_right = [680, 450]
src_bottom_right = [1050, 690]
src_bottom_left = [250, 690]

dest_top_left = [src_bottom_left[0] + (src_top_left[0] - src_bottom_left[0]) / 2, 0]
dest_top_right = [src_top_right[0] + (src_bottom_right[0] - src_top_right[0]) / 2, 0]
dest_bottom_right = [src_top_right[0] + (src_bottom_right[0] - src_top_right[0]) / 2, 720]
dest_bottom_left = [src_bottom_left[0] + (src_top_left[0] - src_bottom_left[0]) / 2, 720]

src_warp = np.float32([src_top_left, src_top_right, src_bottom_right, src_bottom_left])
dest_warp = np.float32([dest_top_left, dest_top_right, dest_bottom_right, dest_bottom_left])
warp_matrix = cv2.getPerspectiveTransform(src_warp, dest_warp)

image = cv2.imread('./test_images/test1.jpg')
image_size = (image.shape[1], image.shape[0])


# =============================================================================
# Processing pipeline
# =============================================================================
undist = cv2.undistort(image, mtx, dist, None, mtx)
warped = cv2.warpPerspective(undist, warp_matrix, image_size, flags=cv2.INTER_LINEAR)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image[..., ::-1])
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warped[..., ::-1])
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()
