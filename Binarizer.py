"""Binarizer"""
import numpy as np
import cv2


class Binarizer():
    """Binarizer - processes an image"""

    def __init__(self):
        """Initialize member variables"""
        self.sobel_kernel = 9
        self.absolute_sobel_threshold = (20, 100)
        self.gradient_magnitude_threshold = (30, 100)
        self.gradient_direction_threshold = (0.7, 1.3)
        self.sobelx = None
        self.sobely = None

    def process(self, image):
        """Process image and output a binarized version"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        self.sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        S = hls[:, :, 2]

        gradx = self.abs_sobel_threshold(gray, orient='x')
        grady = self.abs_sobel_threshold(gray, orient='y', thresh=self.absolute_sobel_threshold)
        mag_binary = self.mag_threshold(S, thresh=self.gradient_magnitude_threshold)
        dir_binary = self.dir_threshold(S, thresh=self.gradient_direction_threshold)

        # Combine all thresholding functions
        combined = np.zeros_like(gray)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined

    def process2(self, image):
        """Process image and output a binarized version"""
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        Y = yuv[:, :, 0]
        U = yuv[:, :, 1]
        V = yuv[:, :, 2]

        Y_binary = np.zeros_like(Y)
        Y_binary[(Y >= 150) & (Y <= 255)] = 1

        U_binary = np.zeros_like(U)
        U_binary[(U >= 100) & (U <= 255)] = 1

        V_binary = np.zeros_like(V)
        V_binary[(V >= 0) & (V <= 100)] = 1

        combined = np.zeros_like(Y)
        combined[(Y_binary == 1) | ((U_binary == 1) & (V_binary == 1))] = 1

        return combined

    def abs_sobel_threshold(self, img, orient='x', thresh=(0, 255)):
        """Apply sobel filter and thresholding"""
        if orient == 'x':
            abs_sobel = np.absolute(self.sobelx)
        if orient == 'y':
            abs_sobel = np.absolute(self.sobely)

        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary

    def mag_threshold(self, img, thresh=(0, 255)):
        """Apply sobel filter and magnitude thresholding"""
        gradmag = np.sqrt(self.sobelx ** 2 + self.sobely ** 2)

        scaled_gradmag = np.uint8(255 * gradmag / np.max(gradmag))

        mag_binary = np.zeros_like(gradmag)
        mag_binary[(scaled_gradmag >= thresh[0]) & (scaled_gradmag <= thresh[1])] = 1

        return mag_binary

    def dir_threshold(self, img, thresh=(0, np.pi / 2)):
        """Apply sobel filter and angle thresholding"""
        absgraddir = np.arctan2(np.absolute(self.sobely), np.absolute(self.sobelx))

        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return dir_binary
