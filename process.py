"""Process images based on the defined pipeline"""
import cv2
import matplotlib.pyplot as plt

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
    binarized = binarizer.process(undist)
    warped = warper.warp(binarized)

    lane_line_image = lane.track_lane_lines(warped)
    newwarp = warper.unWarp(lane_line_image)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)



    
    # Draw black header area
    cv2.rectangle(result, (0, 0), (image.shape[1], 200), (0,0,0), -1)

    # Draw the binarized image
    binary_mini = cv2.resize(binarized, (350, 200), None, 0, 0, cv2.INTER_LINEAR)
    binary_mini = cv2.cvtColor(binary_mini * 255, cv2.COLOR_GRAY2RGB)
    drawx = 1280 - binary_mini.shape[1]
    result[0:0 + binary_mini.shape[0], drawx:drawx + binary_mini.shape[1]] = binary_mini

    # Draw the warped biniarized image combined with the detected lane line
    lane_line_mini = cv2.resize(lane_line_image, (350, 200), None, 0, 0, cv2.INTER_LINEAR)
    warped_mini = cv2.resize(warped, (350, 200), None, 0, 0, cv2.INTER_LINEAR)

    warped_mini = cv2.cvtColor(warped_mini * 255, cv2.COLOR_GRAY2RGB)

    combined_warp_line_mini = cv2.addWeighted(warped_mini, 1, lane_line_mini, 0.3, 0)
    drawx = 1280 - combined_warp_line_mini.shape[1] - 350 - 50
    result[0:0 + combined_warp_line_mini.shape[0], drawx:drawx + combined_warp_line_mini.shape[1]] = combined_warp_line_mini

    # Draw text with lane curvature on current frame
    lane_curvature_text = 'Lane radius: ' + str(round(lane.left_line.radius_of_curvature, 2)) + ' m'
    vehicle_distance_text = 'Distance to lane center: ' + str(round(lane.left_line.line_base_pos, 2)) + ' m'
    cv2.putText(result, lane_curvature_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, vehicle_distance_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result

#image = cv2.imread('./test_images/test7.png')

#output = process_image(image)

#plt.imshow(output[..., ::-1])
#plt.show()

# =============================================================================
# Process video file
# =============================================================================
clip1 = VideoFileClip(input_video_name)
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(output_video_name, audio=False)
