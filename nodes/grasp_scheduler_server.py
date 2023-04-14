#!/usr/bin/env python

import rospy
import smach
import smach_ros

from states import *


# TODO grcnn to common grasprequest


class GraspSchedulerServer:
    def __init__(self) -> None:
        sm = smach.StateMachine(outcomes=["successed", "aborted"])
        sm.userdata.perfer_type=[1,2,3]
        self.sis = smach_ros.IntrospectionServer(
            "grasp_planner_server", sm, "/SM_ROOT"
        )
        self.sis.start()
        with sm:
            smach.StateMachine.add(
                "Init", Init(), transitions={"found": "LookDown", "not_found": "Init"}
            )
            smach.StateMachine.add(
                "LookDown",
                LookDown(),
                transitions={"enough": "Pnp", "not_enough": "LookSide"},
            )
            smach.StateMachine.add(
                "LookSide",
                LookSide(),
                transitions={"enough": "Pnp", "not_enough": "LookSide"},
            )
            smach.StateMachine.add(
                "Pnp",
                Pnp(),
                transitions={"sucessed": "FindChange", "failed": "FailHandler"},
            )
            smach.StateMachine.add(
                "FindChange",
                FindChange(),
                transitions={"change": "LookDown", "not_change": "AsIsPnp","empty":"successed"},
            )
            smach.StateMachine.add(
                "AsIsPnp",
                AsIsPnp(),
                transitions={"successed": "FindChange", "failed": "FailHandler"},
            )
            smach.StateMachine.add(
                "FailHandler",
                FailHandler(),
                transitions={"far": "RetryPnp", "near": "LookDown","failed":"aborted"},
            )
            smach.StateMachine.add(
                "RetryPnp",
                RetryPnp(),
                transitions={"successed": "FindChange", "failed": "FailHandler"},
            )
        outcome = sm.execute()
        rospy.loginfo(f"outcome: {outcome}")
        rospy.signal_shutdown("状态机退出")

    def __del__(self):
        self.sis.stop()


if __name__ == "__main__":
    rospy.init_node("grasp_scheduler_server")
    s = GraspSchedulerServer()
