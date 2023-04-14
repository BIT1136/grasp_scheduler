from collections.abc import Callable
from typing import Any

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Empty

from cam_intri import CameraIntrinsic
from maskrcnn_ros.srv import InstanceSeg,InstanceSegResponse
from grcnn.srv import PredictGraspsWithSeg,PredictGraspsWithSegResponse
from vgn.srv import PredictGrasps,PredictGraspsResponse
from e05_moveit.srv import PickAndPlace,PickAndPlaceResponse

class ROSInterface:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.seg:Callable[[Image],InstanceSegResponse]=rospy.ServiceProxy(
                "/maskrcnn_ros_server/seg_instance", InstanceSeg
            )
        self.grcnn:Callable[[Image,Image,Image],PredictGraspsWithSegResponse] = rospy.ServiceProxy(
                "grcnn_server/predict_grasps", PredictGraspsWithSeg
            )
        self.vgn:Callable[[],PredictGraspsResponse] = rospy.ServiceProxy(
                "vgn_server/predict_grasps", PredictGrasps)
        self.move_cam:Callable[[],None] = rospy.ServiceProxy(
                "move_cam", Empty)
        self.pnp:Callable[[Any],PickAndPlaceResponse] = rospy.ServiceProxy("pick_and_place", PickAndPlace)

        msg: CameraInfo = rospy.wait_for_message(
            "/d435/camera/depth/camera_info", CameraInfo, 1
        )
        self.intrinsic = CameraIntrinsic.from_camera_info(msg)

        self.depth_publisher = rospy.Publisher(
            "/d435/camera/depth/image_convert", Image, queue_size=1)
        