from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

import rospy
import ros_numpy
import tf2_ros
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Transform,TransformStamped

from cam_intri import CameraIntrinsic
from maskrcnn_ros.srv import InstanceSeg,InstanceSegResponse
from grcnn.srv import PredictGraspsWithSeg,PredictGraspsWithSegResponse
from vgn.srv import Integrate,IntegrateResponse
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
        self.vgn_integrate:Callable[[Transform,Image],IntegrateResponse] = rospy.ServiceProxy(
                "vgn_server/integrate", Integrate)
        self.vgn_predict:Callable[[],PredictGraspsResponse] = rospy.ServiceProxy(
                "vgn_server/predict_grasps", PredictGrasps)
        self.move_cam:Callable[[Any],None] = rospy.ServiceProxy(
                "move_cam", Empty)
        self.pnp:Callable[[Any],PickAndPlaceResponse] = rospy.ServiceProxy("pick_and_place", PickAndPlace)

        msg: CameraInfo = rospy.wait_for_message(
            "/d435/camera/depth/camera_info", CameraInfo, 1
        )
        self.intrinsic = CameraIntrinsic.from_camera_info(msg)

        self.depth_publisher = rospy.Publisher(
            "/d435/camera/depth/image_convert", Image, queue_size=1)
        
        self._buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buffer)
        self._broadcaster = tf2_ros.TransformBroadcaster()

    @staticmethod
    def intercept_image()->tuple[Image,Image,npt.NDArray[np.uint8],npt.NDArray[np.uint16]]:
        rgb_msg=rospy.wait_for_message("/d435/camera/color/image_raw", Image, 1)
        depth_msg=rospy.wait_for_message("/d435/camera/depth/image_raw", Image, 1)
        rgb_np=ros_numpy.numpify(rgb_msg)
        depth_np=ros_numpy.numpify(depth_msg)
        return rgb_msg,depth_msg,rgb_np,depth_np

    
    def lookup(self,target_frame, source_frame, time=rospy.Time(0), timeout=rospy.Duration(0))->Transform:
        msg = self._buffer.lookup_transform(target_frame, source_frame, time, timeout)
        return msg.transform
    
    def integrate_tsdf(self,depth_msg):
        extrinsic=self.lookup("depth","tsdf_base")
        self.vgn_integrate(extrinsic,depth_msg)

    def pub_tsdf_base(self,tsdf_base_point):
        # world_to_tsdf_base=np.eye(4)
        # world_to_tsdf_base[0,3]=tsdf_base_point.x
        # world_to_tsdf_base[1,3]=tsdf_base_point.y
        # world_to_tsdf_base[2,3]=tsdf_base_point.z
        # self.world_to_tsdf_base=world_to_tsdf_base

        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "tsdf_base"

        static_transformStamped.transform.translation.x = tsdf_base_point.x
        static_transformStamped.transform.translation.y = tsdf_base_point.y
        static_transformStamped.transform.translation.z = tsdf_base_point.z
        static_transformStamped.transform.rotation.x = 0
        static_transformStamped.transform.rotation.y = 0
        static_transformStamped.transform.rotation.z = 0
        static_transformStamped.transform.rotation.w = 1

        self._broadcaster.sendTransform(static_transformStamped)

    def transform_matrix(self,tf_msg):
        # 将 ROS 的 Transform 消息转换为变换矩阵
        translation = np.array([tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z])
        rotation = np.array([tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z, tf_msg.rotation.w])
        rot=Rotation.from_quat(rotation)
        rotation_matrix = rot.as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix[:3, :3]
        transform_matrix[:3, 3] = translation
        return transform_matrix

    def cam_point_to_world(self,point):
        transform=self.lookup("world","depth",rospy.Time(0),rospy.Duration(0))
        transform_matrix =self.transform_matrix(transform)
        p_c = np.array([point.x, point.y, point.z, 1])
        p_w = np.dot(transform_matrix, p_c)
        return p_w
    
    # def tsdf_to_camera_transform(self, world_to_camera_transform_msg):
    #     # 将从世界坐标系到 tsdf 基坐标系的变换转换为从 tsdf 基坐标系到相机坐标系的变换
    #     world_to_camera_transform = self.transform_matrix(world_to_camera_transform_msg)
    #     tsdf_to_camera_transform = np.dot(world_to_camera_transform, self.world_to_tsdf_base)
    #     tsdf_to_camera_transform_msg = Transform()
    #     tsdf_to_camera_transform_msg.translation.x = tsdf_to_camera_transform[0, 3]
    #     tsdf_to_camera_transform_msg.translation.y = tsdf_to_camera_transform[1, 3]
    #     tsdf_to_camera_transform_msg.translation.z = tsdf_to_camera_transform[2, 3]
    #     rot=Rotation.from_matrix(tsdf_to_camera_transform[:3, :3]).as_quat()
    #     tsdf_to_camera_transform_msg.rotation.x=rot[0]
    #     tsdf_to_camera_transform_msg.rotation.y=rot[1]
    #     tsdf_to_camera_transform_msg.rotation.z=rot[2]
    #     tsdf_to_camera_transform_msg.rotation.w=rot[3]
    #     return tsdf_to_camera_transform_msg