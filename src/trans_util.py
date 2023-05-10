import ros_numpy
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Transform, TransformStamped, Pose


def transform_msg_to_matrix(tf_msg: Transform):
    """将 ROS 的 Transform 消息转换为变换矩阵"""
    translation = np.array(
        [tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z]
    )
    rotation = np.array(
        [tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z, tf_msg.rotation.w]
    )
    # from_quat(x, y, z, w)
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix[:3, :3]
    transform_matrix[:3, 3] = translation
    return transform_matrix


def pose_msg_to_matrix(pose_msg: Pose):
    """将 ROS 的 Pose 消息转换为变换矩阵"""
    translation = np.array(
        [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    )
    rotation = np.array(
        [
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ]
    )
    # from_quat(x, y, z, w)
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix[:3, :3]
    transform_matrix[:3, 3] = translation
    return transform_matrix


def matrix_to_transform_msg(tf_mat):
    """将变换矩阵转换为 ROS 的 Transform 消息"""
    transform = Transform()
    # as_quat()->(x, y, z, w)
    translation = Rotation.from_matrix(tf_mat[:3, :3]).as_quat()
    transform.rotation.x = translation[0]
    transform.rotation.y = translation[1]
    transform.rotation.z = translation[2]
    transform.rotation.w = translation[3]
    transform.translation.x = tf_mat[0, 3]
    transform.translation.y = tf_mat[1, 3]
    transform.translation.z = tf_mat[2, 3]
    return transform


def matrix_to_pose_msg(tf_mat):
    """将变换矩阵转换为 ROS 的 Transform 消息"""
    transform = Pose()
    # as_quat()->(x, y, z, w)
    translation = Rotation.from_matrix(tf_mat[:3, :3]).as_quat()
    transform.orientation.x = translation[0]
    transform.orientation.y = translation[1]
    transform.orientation.z = translation[2]
    transform.orientation.w = translation[3]
    transform.position.x = tf_mat[0, 3]
    transform.position.y = tf_mat[1, 3]
    transform.position.z = tf_mat[2, 3]
    return transform


def apply_trans_to_pose(transform: Transform, pose: Pose) -> Pose:
    t_m = transform_msg_to_matrix(transform)
    p_m = pose_msg_to_matrix(pose)
    np_m = np.dot(t_m, p_m)
    return matrix_to_pose_msg(np_m)


def forward_pose(pose, length) -> Pose:
    pose_mat = pose_msg_to_matrix(pose)
    point = np.dot(pose_mat, np.array([0.0, 0.0, length, 1]))
    new_pose_mat = pose_mat.copy()
    new_pose_mat[:3, 3] = point[:3]
    return matrix_to_pose_msg(new_pose_mat)
