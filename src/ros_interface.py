from collections.abc import Callable
import time
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

import rospy
import ros_numpy
import tf2_ros
from std_msgs.msg import Empty, ColorRGBA
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Transform, TransformStamped, Pose, Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker

from cam_intri import CameraIntrinsic
import trans_util
from maskrcnn_ros.srv import InstanceSeg, InstanceSegResponse
from grcnn.srv import PredictGrasps, PredictGraspsResponse
from vgn.srv import Integrate, IntegrateResponse
from vgn.srv import PredictGrasps, PredictGraspsResponse
from e05_moveit.srv import PickAndPlace, PickAndPlaceResponse
from e05_moveit.srv import MoveCam, MoveCamResponse


class ROSInterface:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance.init = False
        return cls.instance

    def __init__(self):
        if self.init:
            return
        self.init = True

        self.sim = True

        self.seg: Callable[[Image], InstanceSegResponse] = rospy.ServiceProxy(
            "/maskrcnn_ros_server/seg_instance", InstanceSeg
        )
        self.grcnn: Callable[
            [Image, Image, Image], PredictGraspsResponse
        ] = rospy.ServiceProxy("grcnn_server/predict_grasps", PredictGrasps)
        self.vgn_integrate: Callable[
            [Transform, Image], IntegrateResponse
        ] = rospy.ServiceProxy("vgn_server/integrate", Integrate)
        self.vgn_predict: Callable[[], PredictGraspsResponse] = rospy.ServiceProxy(
            "vgn_server/predict_grasps", PredictGrasps
        )
        self.move_cam: Callable[[Pose], MoveCamResponse] = rospy.ServiceProxy(
            "move_cam", MoveCam
        )
        self.pnp: Callable[[Pose], PickAndPlaceResponse] = rospy.ServiceProxy(
            "pick_and_place", PickAndPlace
        )

        self.marker_pub = rospy.Publisher(
            "grasp_scheduler_marker", MarkerArray, queue_size=1
        )
        self.draw_grasp_id = 0
        self.draw_cam_id = 0

        msg: CameraInfo = rospy.wait_for_message(
            "/d435/camera/depth/camera_info", CameraInfo, 1
        )
        self.intrinsic = CameraIntrinsic.from_camera_info(msg)

        self.tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

    def set_sim(self, bool) -> None:
        self.sim = bool

    @staticmethod
    def intercept_image() -> (
        tuple[Image, Image, npt.NDArray[np.uint8], npt.NDArray[np.uint16]]
    ):
        rgb_msg = rospy.wait_for_message("/d435/camera/color/image_raw", Image, 1)
        depth_msg = rospy.wait_for_message("/d435/camera/depth/image_raw", Image, 1)
        rgb_np = ros_numpy.numpify(rgb_msg)
        depth_np = ros_numpy.numpify(depth_msg)
        return rgb_msg, depth_msg, rgb_np, depth_np

    def lookup(self, target_frame, source_frame) -> Transform:
        msg = self.tf_buffer.lookup_transform(
            target_frame, source_frame, rospy.Time().now(), rospy.Duration(1)
        )
        return msg.transform

    def integrate_tsdf(self, depth_msg):
        print("integrating tsdf")
        extrinsic = self.lookup("depth", "tsdf_base")
        self.vgn_integrate(extrinsic, depth_msg)

    def pub_tsdf_base(self, tsdf_cneter_point):
        """发布从elfin_base_link到tsdf_base的坐标变换"""

        # 可能要考虑相机坐标系方向
        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "elfin_base_link"
        static_transformStamped.child_frame_id = "tsdf_base"

        static_transformStamped.transform.translation.x = tsdf_cneter_point[0] - 0.2
        static_transformStamped.transform.translation.y = tsdf_cneter_point[1] - 0.2
        static_transformStamped.transform.translation.z = tsdf_cneter_point[2] - 0.2
        static_transformStamped.transform.rotation.x = 0
        static_transformStamped.transform.rotation.y = 0
        static_transformStamped.transform.rotation.z = 0
        static_transformStamped.transform.rotation.w = 1

        self.tf_broadcaster.sendTransform(static_transformStamped)

    def cam_point_to_base(self, point):
        """计算相机坐标系下的点在elfin_base_link坐标系下的坐标"""
        transform = self.lookup("depth", "elfin_base_link")
        transform_matrix = trans_util.transform_msg_to_matrix(transform)
        p_c = np.array([point[0], point[1], point[2], 1])
        p_w = np.dot(transform_matrix, p_c)
        return p_w[:3]

    def draw_grasp(self, pose: Pose, quality, vmin=0.5, vmax=1.0):
        cm = lambda s: tuple([float(1 - s), float(s), float(0), float(1)])
        color = cm((quality - vmin) / (vmax - vmin))
        markers = create_grasp_markers(
            "elfin_base_link", pose, color, "grasp", self.draw_grasp_id
        )
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.draw_grasp_id += 4

    def draw_cam(self, pose: Pose):
        markers = create_cam_markers("elfin_base_link", pose, "cam", self.draw_cam_id)
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.draw_cam_id += 3

    def clear_markers(self):
        delete = [Marker(action=Marker.DELETEALL)]
        self.marker_pub.publish(MarkerArray(delete))


def create_grasp_markers(frame, pose: Pose, color, ns, id=0):
    pose_mat = trans_util.pose_msg_to_matrix(pose)
    w, d, radius = 0.075, 0.05, 0.005

    left_point = np.dot(pose_mat, np.array([0.0, -w / 2, d / 2, 1]))
    left_pose = pose_mat.copy()
    left_pose[:3, 3] = left_point[:3]
    scale = [radius, radius, d]
    left = create_marker(Marker.CYLINDER, frame, left_pose, scale, color, ns, id)

    right_point = np.dot(pose_mat, np.array([0.0, w / 2, d / 2, 1]))
    right_pose = pose_mat.copy()
    right_pose[:3, 3] = right_point[:3]
    scale = [radius, radius, d]
    right = create_marker(Marker.CYLINDER, frame, right_pose, scale, color, ns, id + 1)

    wrist_point = np.dot(pose_mat, np.array([0.0, 0.0, -d / 4, 1]))
    wrist_pose = pose_mat.copy()
    wrist_pose[:3, 3] = wrist_point[:3]
    scale = [radius, radius, d / 2]
    wrist = create_marker(Marker.CYLINDER, frame, wrist_pose, scale, color, ns, id + 2)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
    palm_pose = np.dot(pose_mat, rot)
    scale = [radius, radius, w]
    palm = create_marker(Marker.CYLINDER, frame, palm_pose, scale, color, ns, id + 3)

    return [left, right, wrist, palm]


def create_cam_markers(frame, pose: Pose, ns="", id=0):
    print("create_cam_markers")
    pose_mat = trans_util.pose_msg_to_matrix(pose)
    # scale.x是轴直径,scale.y是头部直径,scale.z指定头部长度
    arrow_z = create_marker(
        Marker.ARROW, frame, pose_mat, [0.01, 0.02, 0.03], (0, 0, 1, 1), ns, id
    )
    arrow_z.points = [Point(0, 0, 0), Point(0, 0, 0.5)]
    arrow_x = create_marker(
        Marker.ARROW, frame, pose_mat, [0.01, 0.02, 0.03], (1, 0, 0, 1), ns, id + 1
    )
    arrow_x.points = [Point(0, 0, 0), Point(0.1, 0, 0)]
    arrow_y = create_marker(
        Marker.ARROW, frame, pose_mat, [0.01, 0.02, 0.03], (0, 1, 0, 1), ns, id + 2
    )
    arrow_y.points = [Point(0, 0, 0), Point(0, 0.1, 0)]
    return [arrow_z, arrow_x, arrow_y]


def create_marker(
    type, frame, pose: npt.NDArray, scale=[1, 1, 1], color=(1, 1, 1, 1), ns="", id=0
):
    if np.isscalar(scale):
        scale = [scale, scale, scale]
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.ns = ns
    msg.id = id
    msg.type = type
    msg.action = Marker.ADD
    msg.pose = trans_util.matrix_to_pose_msg(pose)
    msg.scale = Vector3(*scale)
    msg.color = ColorRGBA(*color)
    return msg
