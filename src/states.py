import math
import time

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.spatial.transform import Rotation
from skimage import metrics, transform

import rospy
import ros_numpy
import smach
from geometry_msgs.msg import Pose

import grasp_scheduler
import trans_util
from obj_info import ObjectInfo
from ros_interface import ROSInterface
import grcnn.msg, vgn.msg


hold = True
state_start = (
    lambda: None
    if not hold
    else (None if input("continue? [y/N]") == "y" else rospy.signal_shutdown(""))
)


class Init(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["found", "not_found"],
            output_keys=["look_count"],
        )
        rospy.set_param("/place_pose", [0.0, -0.5, 0.02, 0.0, 180.0, 180.0])

    def execute(self, userdata):
        # state_start
        # 为下一个状态初始化数据
        userdata.look_count = 0
        return "found"


class Look(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["enough", "not_enough", "aborted"],
            output_keys=["init_picture", "removed_mask"],
            io_keys=["cam_poses"],
        )

    @staticmethod
    def extract_instance_pointclouds(seg, depth) -> dict[int, npt.NDArray[np.float32]]:
        rospy.logdebug(f"从{seg.max()}个实例中提取点云")
        # print(len(np.where(seg==1)[0]))
        # print(len(np.where(seg==2)[0]))
        # print(len(np.where(seg==3)[0]))
        # 提取实例点云转换到基坐标系下
        rows, cols = seg.shape
        fx = ROSInterface().intrinsic.fx
        fy = ROSInterface().intrinsic.fy
        cx = ROSInterface().intrinsic.cx
        cy = ROSInterface().intrinsic.cy

        # 生成相机坐标系下的网格点
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # 将网格点展开成点云
        points = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)

        instance_pointclouds = {}
        for i in range(rows):
            for j in range(cols):
                instance_id = seg[i, j]
                if instance_id == 0:
                    continue
                if instance_id not in instance_pointclouds:
                    rospy.logdebug(f"发现实例id: {instance_id}")
                    instance_pointclouds[instance_id] = []
                instance_pointclouds[instance_id].append(points[i * cols + j])
        # int->[array(3),array(3),...]

        for instance_id in instance_pointclouds.keys():
            instance_pointclouds[instance_id] = np.array(
                instance_pointclouds[instance_id]
            )
            rospy.logdebug(
                f"实例id={instance_id}点云数量: {len(instance_pointclouds[instance_id])}"
            )
            instance_pointclouds[instance_id] = ROSInterface().cam_pc_to_base(
                instance_pointclouds[instance_id]
            )
        return instance_pointclouds  # int->array(N,3)

    @staticmethod
    def get_object_information():
        rgb_msg, depth_msg, rgb, depth = ROSInterface().intercept_image()
        seg_resp = ROSInterface().seg(rgb_msg)
        seg_out: npt.NDArray[np.uint8] = ros_numpy.numpify(seg_resp.seg)
        instance_pointclouds = __class__.extract_instance_pointclouds(
            seg_out, depth.astype(np.float32) / 1000
        )
        obj_mask_coord = np.where(seg_out != 0)
        obj_mask = np.zeros_like(depth, dtype=bool)
        obj_mask[obj_mask_coord] = True
        rospy.logdebug(f"获取物体信息：分割得到{len(seg_resp.classes)-1}个实例")
        return (
            rgb_msg,
            depth_msg,
            rgb,
            depth,
            seg_resp,
            seg_out,
            instance_pointclouds,
            obj_mask,  # 有物体的像素为True
        )


class LookDown(Look):
    @staticmethod
    def get_roi_center(instance_pointclouds: dict):
        """获取roi的中心点在相机下的坐标"""
        points = []
        for i in instance_pointclouds.values():
            points.append(i)
        points = np.vstack(points)
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        print(min_point, max_point)
        # TODO 移动到桌面上方
        center = (min_point + max_point) / 2
        rospy.logdebug(f"获取roi的中心点在相机下的坐标: {center}")
        return center

    @staticmethod
    def cal_cam_poses(center, count) -> list[Pose]:
        # 定义相机的固定仰角和相机距离
        elevation = math.radians(30)  # 基准点到相机连线与平面法向量的夹角
        distance = 0.5
        z_bias = -0.1

        avoid_arm = 0.5 if count % 2 == 0 else 0

        # 生成等间隔的方位角
        azimuths = [math.radians((i + avoid_arm) * 360 / count) for i in range(count)]

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
            up_norm = up_new / np.linalg.norm(up_new)  # -y
            right_norm = right / np.linalg.norm(right)  # x

            pose_mat = np.eye(4)
            pose_mat[:3, 3] = [x, y, z + z_bias]
            pose_mat[:3, 0] = right_norm
            pose_mat[:3, 1] = -up_norm
            pose_mat[:3, 2] = direction_norm
            pose = trans_util.matrix_to_pose_msg(pose_mat)
            poses.append(pose)
        return poses

    def execute(self, userdata):
        state_start
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
        if len(seg_resp.classes) == 1:
            rospy.logerr("LookDown没有检测到物体,退出")
            return "aborted"
        # np.save("../tests/rgb.npy", rgb)
        userdata.init_picture = rgb
        userdata.removed_mask = np.zeros_like(depth, dtype=bool)  # 已移除物体对应像素为True
        ROSInterface().pub_objmask(np.zeros_like(depth, dtype=bool))
        for i in range(len(instance_pointclouds)):
            ObjectInfo().new_object(i + 1, seg_resp.classes[i + 1])
            mask = np.zeros_like(depth, dtype=bool)
            mask[np.where(seg_out == i + 1)] = True
            ObjectInfo().add_object_mask(i + 1, mask)
            ObjectInfo().add_object_pointcloud(i + 1, instance_pointclouds.get(i + 1))
        pcs = ObjectInfo().get_all_pc()
        ROSInterface().pub_objpc2(pcs)

        resp = ROSInterface().grcnn(rgb_msg, depth_msg, seg_resp.seg)
        grasps: list[grcnn.msg.GraspCandidate] = resp.grasps
        transform = ROSInterface().lookup_cam_to_base()
        for grasp in grasps:
            new_pose = trans_util.apply_trans_to_pose(transform, grasp.pose)
            ObjectInfo().add_object_grasp(grasp.inst_id, new_pose, grasp.quality)
            ROSInterface().draw_grasp(new_pose, grasp.quality)
        rospy.logdebug(f"LookDown后物体信息:\n{ObjectInfo()}")
        pcs = ObjectInfo().get_all_pc()
        ROSInterface().pub_objpc2(pcs)
        # return "enough"#DEBUG
        if depth[objects_mask].max() - depth[objects_mask].min() > 100:
            roi_center = self.get_roi_center(instance_pointclouds)
            rospy.logdebug(f"基坐标系下的tsdf基坐标: {roi_center}")
            ROSInterface().pub_tsdf_base(roi_center, 0.3)
            ROSInterface().draw_roi("tsdf_base", 0.3)
            ObjectInfo().roi_center = roi_center

            ROSInterface().integrate_tsdf(depth_msg)
            userdata.cam_poses = self.cal_cam_poses(roi_center, 2)
            # rospy.logdebug(f"cam poses: {userdata.cam_poses}")
            for cam in userdata.cam_poses:
                ROSInterface().draw_cam(cam)
                time.sleep(0.1)  # 否则有可能不显示
            return "not_enough"
        else:
            return "enough"


class LookSide(Look):
    def execute(self, userdata):
        state_start
        if len(userdata.cam_poses) == 0:
            resp = ROSInterface().vgn_predict()
            grasps: list[vgn.msg.GraspCandidate] = resp.grasps
            # rospy.logdebug(f"vgn grasps: {grasps}")
            transform = ROSInterface().lookup_tsdf_to_base()
            for grasp in grasps:
                # quality=grasp.quality*(0.7+grasp.pose.position.z)#VGN结果衰减
                quality = grasp.quality
                new_pose = trans_util.apply_trans_to_pose(transform, grasp.pose)
                position = ros_numpy.numpify(new_pose.position)
                obj_id = ObjectInfo().find_object_by_position(position)
                if obj_id is None:
                    rospy.logwarn(f"无法找到抓取位置{position}对应的物体,丢弃")
                    continue
                new_pose = trans_util.forward_pose(
                    new_pose, 0.05
                )  # 将夹爪位置沿z轴推进，因为vgn定义的位置在夹爪根部
                new_pose = trans_util.rot_z_90(
                    new_pose
                )  # 将姿势绕z轴旋转90度，因为vgn定义的姿势是右手指为y轴方向
                ObjectInfo().add_object_grasp(obj_id, new_pose, quality)
                ROSInterface().draw_grasp(new_pose, quality)
            rospy.logdebug(f"LookSidePredict后物体信息:\n{ObjectInfo()}")
            pcs = ObjectInfo().get_all_pc()
            ROSInterface().pub_objpc2(pcs)
            return "enough"
        pos = userdata.cam_poses.pop()
        # rospy.logdebug(f"移动相机到: {pos}")
        try:
            resp = ROSInterface().move_cam(pos)
        except Exception as e:
            rospy.logerr(f"调用移动相机服务失败:{e}")
            return "not_enough"
        if resp.result == False:
            rospy.logerr(f"移动相机失败,目标:{pos}")
            return "not_enough"
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
        for i in range(len(seg_resp.classes) - 1):
            if instance_pointclouds.get(i + 1) is None:
                continue
            ObjectInfo().merge_object_pointcloud(
                instance_pointclouds[i + 1], seg_resp.classes[i + 1]
            )
        rospy.logdebug(f"LookSide后物体信息:\n{ObjectInfo()}")
        pcs = ObjectInfo().get_all_pc()
        ROSInterface().pub_objpc2(pcs)
        return "not_enough"


class Pnp(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["successed", "failed", "aborted", "finished"],
            input_keys=["perfer_type"],
            output_keys=["fail_counter", "fail_location"],
            io_keys=["removed_mask"],
        )

    def fine_target(self, perfer_type):
        target_type = 0
        for type in perfer_type:
            if type in ObjectInfo().grasp_type_count:
                if ObjectInfo().grasp_type_count[type] > 0:
                    target_type = type
                    break
        return target_type

    def execute(self, userdata):
        state_start
        target_type = self.fine_target(userdata.perfer_type)
        if target_type == 0:
            return "finished"
        rospy.logdebug(f"抓取目标类型: {target_type}")
        # rospy.set_param("pick_and_place/object_type", target_type)
        grasp, plans, removed_obj_mask = ObjectInfo().get_best_grasp_plan_and_remove(
            target_type
        )
        if grasp is None:
            return "failed"
        rospy.logdebug(f"最佳抓取: {grasp}")
        userdata.removed_mask += removed_obj_mask
        ROSInterface().pub_objmask(userdata.removed_mask)
        try:
            resp = ROSInterface().pnp(grasp)
        except Exception as e:
            rospy.logerr(f"调用抓取服务失败:{e}")
            return "aborted"
        if resp.result:
            rospy.logdebug(f"Pnp后物体信息:\n{ObjectInfo()}")
            if self.fine_target(userdata.perfer_type) == 0:
                return "finished"
            return "successed"
        else:
            rospy.logerr(f"抓取中途掉落")
            userdata.fail_counter = 1
            # userdata.fail_location = resp.fail_location
            return "failed"


class FindChange(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["change", "no_change"],
            input_keys=["init_picture", "removed_mask"],
        )

    def execute(self, userdata):
        rgbmask = (
            np.expand_dims(userdata.removed_mask, axis=2)
            .repeat(3, axis=2)
            .astype(np.uint8)
        )
        init_pic = userdata.init_picture * (1 - rgbmask)
        _, _, pic, _ = ROSInterface().intercept_image()
        pic *= 1 - rgbmask
        # np.save("../tests/init_pic.npy", init_pic)
        # np.save("../tests/pic.npy", pic)
        type = pic.dtype
        init_pic = transform.resize(
            init_pic,
            (200, 200, 3),
            preserve_range=True,
        ).astype(type)
        pic = transform.resize(pic, (200, 200, 3), preserve_range=True).astype(type)
        ssim_value = metrics.structural_similarity(init_pic, pic, channel_axis=2)
        rospy.logdebug(f"ssim值:{ssim_value}，{'改变' if ssim_value < 0.93 else '不变'}")
        if ssim_value < 0.93:
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
        obj = ObjectInfo().roi_center
        obj[2] = 0
        location[2] = 0
        dist = np.linalg.norm(obj - location)
        return dist

    def execute(self, userdata):
        state_start
        if userdata.fail_counter > 3:
            return "failed"
        userdata.fail_counter += 1
        distance = self.calc_fail_distnce(userdata.fail_location)
        if distance > 0.75:
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
        state_start
        # pose=__class__.look_fail_location(userdata.fail_location)
        # ROSInterface().move_cam(pose)
        # ROSInterface().look_down_now()
        rgb_msg, depth_msg, _, _ = ROSInterface().intercept_image()
        seg_resp = ROSInterface().seg(rgb_msg)
        resp = ROSInterface().grcnn(rgb_msg, depth_msg, seg_resp.seg)
        resp = ROSInterface().pnp(resp.grasps[0])
        if resp.result:
            return "successed"
        else:
            userdata.fail_location = resp.fail_location
            return "failed"
