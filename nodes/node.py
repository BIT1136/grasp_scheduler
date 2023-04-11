#!/home/yangzhuo/mambaforge/envs/grasp/bin/python

import rospy

from grasp_planner.msg import mymsg
from grasp_planner.srv import mysrv, mysrvResponse
import utils

if __name__ == "__main__":
    rospy.init_node("grasp_planner_server")
    rospy.spin()
