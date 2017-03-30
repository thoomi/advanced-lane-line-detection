"""Process images based on the defined pipeline"""
import cv2

from Binarizer import Binarizer
from CameraCalibrator import CameraCalibrator
from Lane import Lane
from moviepy.editor import VideoFileClip
from optparse import OptionParser
from Warper import Warper


# =============================================================================
# Get command line arguments
# =============================================================================
parser = OptionParser()
(options, args) = parser.parse_args()

input_video_name = args[0]
output_video_name = 'result_' + input_video_name

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
clip1 = VideoFileClip(input_video_name)
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(output_video_name, audio=False)
