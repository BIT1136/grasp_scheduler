import math

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import rospy
import ros_numpy
import smach
from geometry_msgs.msg import Pose

from type import *
from ssim import SSIM
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
        # 为下一个状态初始化数据
        userdata.retry_count = 0
        userdata.look_count = 0
        return "found"


class Look(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["enough", "not_enough", "bias", "aborted"],
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

        instance_pointclouds = {}
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

    def get_object_information(self):
        rgb_msg, depth_msg, rgb, depth = ROSInterface.intercept_image()
        seg_resp = ROSInterface().seg(rgb_msg)
        seg_out: npt.NDArray[np.uint8] = ros_numpy.numpify(seg_resp.seg)
        instance_pointclouds = __class__.extract_instance_pointclouds(
            seg_out, depth.astype(np.float32) / 1000
        )
        instances_mask = np.where(seg_out != 0)
        return (
            rgb_msg,
            depth_msg,
            rgb,
            depth,
            seg_resp,
            instance_pointclouds,
            instances_mask,
        )

    def _execute(self, userdata):
        raise Exception

    def execute(self, userdata):
        for i in range(3):
            try:
                res = self._execute(userdata)
            except rospy.ROSException as e:
                rospy.logerr(e)
            except rospy.ServiceException as e:
                rospy.logerr(e)
            else:
                return res
        return "aborted"


class LookDown(Look):
    @staticmethod
    def get_roi(points, l):
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        center = (min_point + max_point) / 2
        base_point = center - l / 2
        return base_point

    @staticmethod
    def cal_cam_poses(roi_base, count) -> list[Pose]:
        # 定义相机的固定仰角和相机距离
        elevation = math.radians(30)  # 基准点到相机连线与平面法向量的夹角
        distance = 0.75

        # 生成等间隔的方位角
        azimuths = [math.radians(i * 360 / count) for i in range(count)]

        poses = []
        for azimuth in azimuths:
            x = roi_base[0] + distance * math.sin(elevation) * math.cos(azimuth)
            y = roi_base[1] + distance * math.sin(elevation) * math.sin(azimuth)
            z = roi_base[2] + distance * math.cos(elevation)

            # 计算相机朝向四元数
            direction = [roi_base[i] - coord for i, coord in enumerate([x, y, z])]
            norm = math.sqrt(sum([d**2 for d in direction]))
            if norm > 0:
                direction = [d / norm for d in direction]

            # 计算旋转轴和旋转角度
            v1 = np.array([0, 0, 1])
            v2 = np.array(direction)
            axis = np.cross(v1, v2)
            angle = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))

            # 使用scipy中的Rotation类构造旋转矩阵和四元数
            r = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
            q = r.as_quat()

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            poses.append(pose)
        rospy.logdebug(f"cam move to: {poses}")
        return poses

    def _execute(self, userdata):
        (
            rgb_msg,
            depth_msg,
            rgb,
            depth,
            seg_resp,
            instance_pointclouds,
            instances_mask,
        ) = self.get_object_information()
        userdata.init_picture = rgb
        for i in range(len(instance_pointclouds)):
            ObjectInfo().new_object(i + 1, seg_resp.classes[i + 1])
            mask = np.zeros_like(rgb, dtype=bool)
            mask[instances_mask[i + 1]] = True
            ObjectInfo().add_object_mask(i + 1, mask)
            ObjectInfo().add_object_pointcloud(i + 1, instance_pointclouds[i + 1])
        resp = ROSInterface().grcnn(rgb_msg, depth_msg, seg_resp.seg)
        grasps: list[GraspCandidateWithIdx] = resp.grasps
        for grasp in grasps:
            ObjectInfo().add_object_grasp(grasp.inst_id, grasp.pose, grasp.quality)
        if depth[instances_mask].max() - depth[instances_mask].min() > 100:
            roi_base = __class__.get_roi(instance_pointclouds, 0.4)
            roi_base = ROSInterface().cam_point_to_world(roi_base)
            rospy.logdebug(f"roi base in world: {roi_base}")
            ROSInterface().pub_tsdf_base(roi_base)
            ROSInterface().integrate_tsdf(depth_msg)
            userdata.cam_poses = __class__.cal_cam_poses(roi_base + 0.2, 4)
            ObjectInfo().roi_center = roi_base + 0.2
            return "not_enough"
        else:
            return "enough"


class LookSide(Look):
    def _execute(self, userdata):
        pos = userdata.cam_poses.pop()
        ROSInterface().move_cam(pos)
        (
            _,
            depth_msg,
            _,
            _,
            seg_resp,
            instance_pointclouds,
            _,
        ) = self.get_object_information()
        ROSInterface().integrate_tsdf(depth_msg)
        if len(userdata.cam_poses) == 0:
            for i in range(len(seg_resp.classes)):
                ObjectInfo().add_object_pointcloud(
                    instance_pointclouds[i], seg_resp.classes[i]
                )
            ROSInterface().depth_publisher.publish(depth_msg)
            resp = ROSInterface().vgn_predict()
            grasps: list[GraspCandidate] = resp.grasps
            for grasp in grasps:
                position = ros_numpy.numpify(grasp.pose.position)
                obj_id = ObjectInfo().find_object_by_position(position)
                ObjectInfo().add_object_grasp(obj_id, grasp.pose, grasp.quality)
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
        target_type = 0
        for type in userdata.perfer_type:
            if ObjectInfo().grasp_type_count[type] > 0:
                target_type = type
                break
        grasp, removed_obj_mask = ObjectInfo().get_best_grasp_plan_and_remove(
            target_type
        )
        userdata.obj_mask += removed_obj_mask
        resp = ROSInterface().pnp(grasp)
        if resp.result:
            return "successed"
        else:
            userdata.fail_counter = 1
            userdata.fail_location = resp.fail_location
            return "failed"


class FindChange(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["change", "no_change", "empty"],
            input_keys=["init_picture", "obj_mask"],
        )

    def execute(self, userdata):
        init_pic = userdata.init_picture * (1 - userdata.obj_mask)
        init_pic = Image.fromarray(init_pic, mode="RGB")
        _, _, pic, _ = ROSInterface.intercept_image()
        pic *= 1 - userdata.obj_mask
        now_picture = Image.fromarray(pic, mode="RGB")
        ssim = SSIM(init_pic)
        ssim_value = ssim.cw_ssim_value(now_picture)
        if ssim_value < 0.9:
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
