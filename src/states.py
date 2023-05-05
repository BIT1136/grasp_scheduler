import math
import time

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.spatial.transform import Rotation

import rospy
import ros_numpy
import smach
from geometry_msgs.msg import Pose

import grasp_scheduler
from ssim import SSIM
import trans_util
from obj_info import ObjectInfo
from ros_interface import ROSInterface
from grcnn.msg import GraspCandidateWithIdx
from vgn.msg import GraspCandidate


class Init(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["found", "not_found"],
            output_keys=["retry_count", "look_count"],
        )
        rospy.set_param("/place_pose", [0.0, -0.5, 0.02, 0.0, 180.0, 180.0])

    def execute(self, userdata):
        None if input("continue? [y/N]") == "y" else rospy.signal_shutdown("")
        # 为下一个状态初始化数据
        userdata.retry_count = 0
        userdata.look_count = 0
        return "found"


class Look(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["enough", "not_enough", "aborted"],
            output_keys=["init_picture"],
            io_keys=["retry_count", "cam_poses"],
        )

    @staticmethod
    def extract_instance_pointclouds(seg, depth):
        rows, cols = seg.shape
        pixel_indices = np.indices((rows, cols)).reshape(2, -1)
        pixel_indices = np.concatenate(
            (pixel_indices, np.ones((1, pixel_indices.shape[1]))), axis=0
        )
        points = (
            np.linalg.inv(ROSInterface().intrinsic.as_matrix())
            @ pixel_indices
            * depth.reshape(1, -1)
        )
        points = points[:3, :].T

        instance_pointclouds = {}  # int->array(N,3)
        for i in range(rows):
            for j in range(cols):
                instance_id = seg[i, j]
                if instance_id == 0:
                    continue
                if instance_id not in instance_pointclouds:
                    instance_pointclouds[instance_id] = []
                instance_pointclouds[instance_id].append(points[i * cols + j])

        for instance_id in instance_pointclouds:
            instance_pointclouds[instance_id] = np.array(
                instance_pointclouds[instance_id]
            )
        return instance_pointclouds

    @staticmethod
    def get_object_information():
        rgb_msg, depth_msg, rgb, depth = ROSInterface.intercept_image()
        seg_resp = ROSInterface().seg(rgb_msg)
        seg_out: npt.NDArray[np.uint8] = ros_numpy.numpify(seg_resp.seg)
        instance_pointclouds = __class__.extract_instance_pointclouds(
            seg_out, depth.astype(np.float32) / 1000
        )
        objects_mask = np.where(seg_out != 0)
        return (
            rgb_msg,
            depth_msg,
            rgb,
            depth,
            seg_resp,
            seg_out,
            instance_pointclouds,
            objects_mask,
        )

    def _execute(self, userdata):
        raise Exception

    def execute(self, userdata):
        None if input("continue? [y/N]") == "y" else rospy.signal_shutdown("")
        for _ in range(1):
            try:
                res = self._execute(userdata)
            except rospy.ROSException as e:
                rospy.logerr(e)
            except rospy.ServiceException as e:
                rospy.logerr(e)
            else:
                rospy.logdebug(
                    f"obj after look, dict: {ObjectInfo().obj_dict} count: {ObjectInfo().obj_dict}"
                )
                return res
        return "aborted"


class LookDown(Look):
    @staticmethod
    def get_roi_center(instance_pointclouds: dict):
        """获取roi的中心点在depth下的坐标"""
        points = []
        for i in instance_pointclouds.values():
            points.append(i)
        points = np.vstack(points)
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        # TODO 移动到桌面上方
        center = (min_point + max_point) / 2
        print(f"roi_center_point: {center}")
        return center

    @staticmethod
    def cal_cam_poses(center, count) -> list[Pose]:
        # 定义相机的固定仰角和相机距离
        elevation = math.radians(30)  # 基准点到相机连线与平面法向量的夹角
        distance = 0.75

        # 生成等间隔的方位角
        azimuths = [math.radians(i * 360 / count) for i in range(count)]

        poses = []
        for azimuth in azimuths:
            x = center[0] + distance * math.sin(elevation) * math.cos(azimuth)
            y = center[1] + distance * math.sin(elevation) * math.sin(azimuth)
            z = center[2] + distance * math.cos(elevation)

            direction = np.array([center[0] - x, center[1] - y, center[2] - z])
            up = np.array([0, 0, 1])
            right = np.cross(direction, up)
            up_new = np.cross(right, direction)
            direction_norm = direction / np.linalg.norm(direction)  # z
            right_norm = right / np.linalg.norm(right)  # y
            up_norm = up_new / np.linalg.norm(up_new)  # x

            pose_mat = np.eye(4)
            pose_mat[:3, 3] = [x, y, z]
            pose_mat[:3, 0] = up_norm
            pose_mat[:3, 1] = right_norm
            pose_mat[:3, 2] = direction_norm
            pose = trans_util.matrix_to_pose_msg(pose_mat)
            poses.append(pose)
        rospy.logdebug(f"cam poses to: {poses}")
        return poses

    def _execute(self, userdata):
        (
            rgb_msg,
            depth_msg,
            rgb,
            depth,
            seg_resp,
            seg_out,
            instance_pointclouds,
            objects_mask,
        ) = self.get_object_information()
        userdata.init_picture = rgb
        for i in range(len(instance_pointclouds)):
            ObjectInfo().new_object(i + 1, seg_resp.classes[i + 1])
            mask = np.zeros_like(rgb, dtype=bool)
            mask[np.where(seg_out == i + 1)] = True
            ObjectInfo().add_object_mask(i + 1, mask)
            ObjectInfo().add_object_pointcloud(i + 1, instance_pointclouds[i + 1])

        resp = ROSInterface().grcnn(rgb_msg, depth_msg, seg_resp.seg)
        grasps: list[GraspCandidateWithIdx] = resp.grasps
        rospy.logdebug(f"grcnn grasps in depth: {grasps}")
        transform = ROSInterface().lookup("depth", "elfin_base_link")
        for grasp in grasps:
            new_pose = trans_util.apply_trans_to_pose(transform, grasp.pose)
            ObjectInfo().add_object_grasp(grasp.inst_id, new_pose, grasp.quality)
            ROSInterface().draw_grasp(new_pose, grasp.quality)
        # return "enough"#DEBUG
        if depth[objects_mask].max() - depth[objects_mask].min() > 100:
            roi_center = self.get_roi_center(instance_pointclouds)
            roi_center = ROSInterface().cam_point_to_base(roi_center)
            rospy.logdebug(f"roi base in elfin_base_link: {roi_center}")
            ROSInterface().pub_tsdf_base(roi_center)
            ObjectInfo().roi_center = roi_center

            ROSInterface().integrate_tsdf(depth_msg)
            userdata.cam_poses = self.cal_cam_poses(roi_center, 4)
            rospy.logdebug(f"cam poses: {userdata.cam_poses}")
            for cam in userdata.cam_poses:
                ROSInterface().draw_cam(cam)
            return "not_enough"
        else:
            return "enough"


class LookSide(Look):
    def _execute(self, userdata):
        pos = userdata.cam_poses.pop()
        rospy.logdebug(f"move cam to: {pos}")
        ROSInterface().move_cam(pos)
        (
            _,
            depth_msg,
            _,
            _,
            seg_resp,
            _,
            instance_pointclouds,
            _,
        ) = self.get_object_information()
        ROSInterface().integrate_tsdf(depth_msg)
        for i in range(len(seg_resp.classes)):
            ObjectInfo().merge_object_pointcloud(
                instance_pointclouds[i], seg_resp.classes[i]
            )
        if len(userdata.cam_poses) == 0:
            resp = ROSInterface().vgn_predict()
            grasps: list[GraspCandidate] = resp.grasps
            rospy.logdebug(f"vgn grasps: {grasps}")
            transform = ROSInterface().lookup("tsdf_base", "elfin_base_link")
            for grasp in grasps:
                position = ros_numpy.numpify(grasp.pose.position)
                obj_id = ObjectInfo().find_object_by_position(position)
                new_pose = trans_util.apply_trans_to_pose(transform, grasp.pose)
                ObjectInfo().add_object_grasp(obj_id, new_pose, grasp.quality)
                ROSInterface().draw_grasp(new_pose, grasp.quality)
            return "enough"
        return "not_enough"


class Pnp(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["successed", "failed"],
            output_keys=["obj_mask", "fail_counter", "fail_location"],
        )

    def execute(self, userdata):
        None if input("continue? [y/N]") == "y" else rospy.signal_shutdown("")
        target_type = 0
        for type in userdata.perfer_type:
            if ObjectInfo().grasp_type_count[type] > 0:
                target_type = type
                break
        rospy.logdebug(f"target_type: {target_type}")
        rospy.set_param("pick_and_place/object_type", target_type)
        grasp, removed_obj_mask = ObjectInfo().get_best_grasp_plan_and_remove(
            target_type
        )
        if grasp is None:
            return "failed"
        rospy.logdebug(f"grasp: {grasp}")
        userdata.obj_mask += removed_obj_mask
        rospy.logdebug(f"userdata.obj_mask: {userdata.obj_mask}")
        resp = ROSInterface().pnp(grasp)
        if resp.result:
            return "successed"
        else:
            userdata.fail_counter = 1
            # userdata.fail_location = resp.fail_location
            return "failed"


class FindChange(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["change", "no_change", "empty"],
            input_keys=["init_picture", "obj_mask"],
        )

    def execute(self, userdata):
        None if input("continue? [y/N]") == "y" else rospy.signal_shutdown("")
        init_pic = userdata.init_picture * (1 - userdata.obj_mask)
        init_pic = Image.fromarray(init_pic, mode="RGB")
        _, _, pic, _ = ROSInterface.intercept_image()
        pic *= 1 - userdata.obj_mask
        now_picture = Image.fromarray(pic, mode="RGB")
        ssim = SSIM(init_pic)
        ssim_value = ssim.cw_ssim_value(now_picture)
        rospy.logdebug(f"ssim value: {ssim_value}")
        if ssim_value < 0.8:
            ROSInterface().clear_markers()
            return "change"
        else:
            return "no_change"


class FailHandler(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["far", "near", "failed"],
            io_keys=["fail_counter", "fail_location"],
        )

    @staticmethod
    def calc_fail_distnce(location):
        ObjectInfo().roi_center
        location
        return 0

    def execute(self, userdata):
        None if input("continue? [y/N]") == "y" else rospy.signal_shutdown("")
        if userdata.fail_counter > 3:
            return "failed"
        userdata.fail_counter += 1
        distance = __class__.calc_fail_distnce(userdata.fail_location)
        if distance > 100:
            return "far"
        else:
            return "near"


class RetryPnp(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["successed", "failed"],
            input_keys=["fail_location"],
            output_keys=["fail_counter"],
        )

    # @staticmethod
    # def look_fail_location(fail_location):
    #     pose=fail_location
    #     pose.position.z+=1
    #     r = Rotation.from_rotvec([math.pi,0,0])#RetryPnp向下看位姿：失败位置向上1m，z轴向下
    #     q = r.as_quat()
    #     pose.orientation.x = q[0]
    #     pose.orientation.y = q[1]
    #     pose.orientation.z = q[2]
    #     pose.orientation.w = q[3]
    #     return pose

    def execute(self, userdata):
        None if input("continue? [y/N]") == "y" else rospy.signal_shutdown("")
        # pose=__class__.look_fail_location(userdata.fail_location)
        # ROSInterface().move_cam(pose)
        ROSInterface().look_down_now()
        rgb_msg, depth_msg, _, _ = ROSInterface.intercept_image()
        seg_resp = ROSInterface().seg(rgb_msg)
        resp = ROSInterface().grcnn(rgb_msg, depth_msg, seg_resp.seg)
        resp = ROSInterface().pnp(resp.grasps[0])
        if resp.result:
            return "successed"
        else:
            userdata.fail_location = resp.fail_location
            return "failed"
