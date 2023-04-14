import numpy as np

import rospy
import ros_numpy
import smach
import smach_ros


from type import *
import utils
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
            outcomes=["enough", "not_enough", "bias"],
            output_keys=["picture"],
            io_keys=["retry_count", "look_count","cam_poses"],
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
        self.rgb_msg, self.depth_msg, _, self.depth = utils.intercept_image()
        self.depth=self.depth.astype(np.float32)/1000
        self.seg_resp = ROSInterface().seg(self.rgb_msg)
        seg_out: npt.NDArray[np.uint8] = ros_numpy.numpify(self.seg_resp.seg)
        self.instance_pointclouds = self.__class__.extract_instance_pointclouds(
            seg_out, self.depth
        )
        self.instances_mask = np.where(seg_out != 0)

    def _execute(self, userdata):
        raise Exception

    def execute(self, userdata):
        try:
            res = self._execute(userdata)
        except rospy.ROSException as e:
            rospy.logerr(e)
        except rospy.ServiceException as e:
            rospy.logerr(e)
        else:
            return res


class LookDown(Look):
    def _execute(self, userdata):
        self.get_object_information()
        ros_base=utils.get_roi(self.instance_pointclouds,0.4)
        for i in range(len(self.instance_pointclouds)):
            ObjectInfo().new_object(i + 1, self.seg_resp.classes[i])
            ObjectInfo().add_object_pointcloud(
                self.instance_pointclouds[i + 1], self.seg_resp.classes[i + 1]
            )
        resp = ROSInterface().grcnn(self.rgb_msg, self.depth_msg, self.seg_resp.seg)
        grasps: list[GraspCandidateWithIdx] = resp.grasps
        for grasp in grasps:
            ObjectInfo().add_object_grasp(grasp.inst_id, grasp.pose, grasp.quality)
        if (
            self.depth[self.instances_mask].max()
            - self.depth[self.instances_mask].min()
            < 0.1
        ):
            return "enough"
        else:
            return "not_enough"


class LookSide(Look):
    def _execute(self, userdata):
        # userdata.look_count += 1
        pos=userdata.cam_poses.pop()
        # TODO get pos
        ROSInterface().move_cam(pos)
        self.get_object_information()
        # if userdata.look_count == 3:
        if len(userdata.cam_poses)==0:
            for i in range(len(self.seg_resp.classes)):
                ObjectInfo().add_object_pointcloud(
                    self.instance_pointclouds[i], self.seg_resp.classes[i]
                )
            ROSInterface().depth_publisher.publish(self.depth_msg)
            resp = ROSInterface().vgn()
            grasps: list[GraspCandidate] = resp.grasps
            for grasp in grasps:
                position = ros_numpy.numpify(grasp.pose.position)
                obj_id = ObjectInfo().find_object_by_position(position)
                ObjectInfo().add_object_grasp(obj_id, grasp.pose, grasp.quality)
            return "enough"
        return "not_enough"


class Pnp(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["successed", "failed"],output_keys=["obj_mask","fail_counter"])

    def execute(self, userdata):
        target_type = 0
        for type in userdata.perfer_type:
            if ObjectInfo().grasp_type_count[type] > 0:
                target_type = type
                break
        grasp = ObjectInfo().get_best_grasp_plan_and_remove(target_type)
        resp = ROSInterface().pnp(grasp)
        if resp.result:
            userdata.obj_mask=resp.mask
            return "successed"
        else:
            userdata.fail_counter=1
            return "failed"


class FindChange(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["change", "no_change","empty"],io_keys=["picture"])

    def execute(self, userdata):
        _, _, pic, _ = utils.intercept_image()
        previous_pic=userdata.picture
        # compare image
        flag = True
        if flag:
            return "change"
        else:
            return "no_change"


class AsIsPnp(Pnp):
    def execute(self, userdata):
        # pick grasp from obj info
        resp = ROSInterface().pnp()
        if resp.result:
            return "successed"
        else:
            return "failed"


class FailHandler(smach.State):
    def __init__(self):
        smach.State.__init__(
            self, outcomes=["far", "near", "failed"], io_keys=["fail_counter"]
        )

    def execute(self, userdata):
        if userdata.fail_counter > 3:
            return "failed"
        userdata.fail_counter+=1
        # compare pnp failed place
        distance = 0
        if distance > 100:
            return "far"
        else:
            return "near"


class RetryPnp(Pnp):
    def execute(self, userdata):
        rgb_msg, depth_msg, _, _ = utils.intercept_image()
        seg_resp = ROSInterface().seg(rgb_msg)
        data = ROSInterface().grcnn(rgb_msg, depth_msg, seg_resp.seg)
        resp = ROSInterface().pnp(data)
        if resp.result:
            return "successed"
        else:
            return "failed"
