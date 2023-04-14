#!/usr/bin/env python

import rospy
import smach
import smach_ros

class A(smach.State):
    def __init__(self):
        smach.State.__init__(self, 
                             outcomes=['outcome1','outcome2'],
                             io_keys=['sm_counter'],
                             input_keys=['list'])

    def execute(self, userdata):
        rospy.loginfo(userdata.sm_counter)
        print(userdata.list)
        userdata.list.pop()
        if userdata.sm_counter < 3:
            userdata.sm_counter = userdata.sm_counter + 1
            return 'outcome1'
        else:
            return 'outcome2'
        
class B(smach.State):
    def __init__(self):
        smach.State.__init__(self, 
                             outcomes=['outcome1'])

    def execute(self, userdata):
        return 'outcome1'

class C(smach.State):
    def __init__(self):
        smach.State.__init__(self, 
                             outcomes=['outcome1'])

    def execute(self, userdata):
        return 'outcome1'

def main():
    rospy.init_node('smach_example_state_machine')

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['outcome4'])
    sm.userdata.sm_counter = 0
    sm.userdata.list=[1,2,3]

    # Open the container
    with sm:
        smach.StateMachine.add('A', A(), 
                               transitions={'outcome1':'B', 
                                            'outcome2':'outcome4'})
        smach.StateMachine.add('B', B(),
                                 transitions={'outcome1':'C'})
        smach.StateMachine.add('C', C(),
                                    transitions={'outcome1':'A'})

    # Execute SMACH plan
    outcome = sm.execute()

if __name__ == '__main__':
    main()
