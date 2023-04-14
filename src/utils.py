import numpy as np

import rospy
import ros_numpy
from sensor_msgs.msg import Image

import grasp_scheduler
from type import *


def intercept_image()->tuple[Image,Image,npt.NDArray[np.uint8],npt.NDArray[np.uint16]]:
    rgb_msg=rospy.wait_for_message("/d435/camera/color/image_raw", Image, 1)
    depth_msg=rospy.wait_for_message("/d435/camera/depth/image_raw", Image, 1)
    rgb_np=ros_numpy.numpify(rgb_msg)
    depth_np=ros_numpy.numpify(depth_msg)
    return rgb_msg,depth_msg,rgb_np,depth_np

def get_roi(points,l):
    min_point=np.min(points,axis=0)
    max_point=np.max(points,axis=0)
    center=(min_point+max_point)/2
    base_point=center-l/2
    return base_point

