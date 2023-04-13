#!/usr/bin/env python

import time

import rospy
import smach
import smach_ros
from sensor_msgs.msg import Image

from grcnn.srv import PredictGrasps
from e05_moveit.srv import PickAndPlace

# TODO grcnn to common grasprequest


class Root(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "aborted", "preempted"])
        rospy.set_param("/place_pose", [0.0, -0.5, 0.02, 0.0, 180.0, 180.0])

    def execute(self, userdata):
        a = input("grasp? [y/N]")
        if a == "y":
            return "succeeded"
        else:
            return "aborted"


class Grasp(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "aborted", "preempted"])

    def execute(self, userdata):
        try:
            rgb = rospy.wait_for_message("/d435/camera/color/image_raw", Image, 1)
            depth = rospy.wait_for_message("/d435/camera/depth/image_raw", Image, 1)
            handle = rospy.ServiceProxy("grcnn_server/predict_grasps", PredictGrasps)
            data = handle(rgb, depth)
            rospy.loginfo(f"grasp: {data.grasps[0]}")
            handle = rospy.ServiceProxy("pick_and_place", PickAndPlace)
            resp = handle(data.grasps[0])
        except Exception as e:
            rospy.logerr(e)
            return "aborted"
        else:
            if resp:
                return "succeeded"
            else:
                rospy.logerr("pnp failed")
                return "aborted"


class GraspSchedulerServer:
    def __init__(self) -> None:
        self.sm = smach.StateMachine(outcomes=["succeeded", "aborted", "preempted"])
        self.sis = smach_ros.IntrospectionServer(
            "grasp_planner_server", self.sm, "/SM_ROOT"
        )
        self.sis.start()
        with self.sm:
            smach.StateMachine.add("root", Root(), transitions={"succeeded": "grasp","aborted":"aborted"})
            smach.StateMachine.add("grasp", Grasp(), transitions={"succeeded": "root","aborted":"aborted"})
        outcome = self.sm.execute()
        rospy.loginfo(f"outcome: {outcome}")
        rospy.signal_shutdown("done")

    def __del__(self):
        self.sis.stop()


if __name__ == "__main__":
    rospy.init_node("grasp_scheduler_server")
    s = GraspSchedulerServer()
