"""Process images based on the defined pipeline"""
import cv2

from Binarizer import Binarizer
from CameraCalibrator import CameraCalibrator
from Lane import Lane
from moviepy.editor import VideoFileClip
from Warper import Warper


# =============================================================================
# Create processing instances
# =============================================================================
calibrator = CameraCalibrator()
binarizer = Binarizer()
warper = Warper()
lane = Lane()

calibrator.loadParameters("./camera_cal/calibration_coefficients.p")


# =============================================================================
# Preprocessing pipeline
# =============================================================================
def process_image(image):
    """Process a single image"""
    undist = calibrator.undistort(image)
    binarized = binarizer.process2(undist)
    warped = warper.warp(binarized)

    lane_line_image = lane.track_lane_lines(warped)
    newwarp = warper.unWarp(lane_line_image)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


# =============================================================================
# Process video file
# =============================================================================
project_output = 'project_video_result.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_output, audio=False)
