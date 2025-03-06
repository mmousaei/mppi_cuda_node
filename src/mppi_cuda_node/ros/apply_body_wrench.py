#!/usr/bin/env python
import rospy
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest
from geometry_msgs.msg import Wrench, Vector3

def apply_force():
    rospy.init_node('apply_wrench_node')
    rospy.wait_for_service('/gazebo/apply_body_wrench')
    try:
        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        
        req = ApplyBodyWrenchRequest()
        # Adjust this to your drone's link name in Gazebo (e.g., "your_drone::base_link")
        req.body_name = "hexa_x_tilt::base_link"
        req.reference_frame = "world"
        
        # Define the wrench: 5 N force in x-direction; no torque.
        req.wrench.force = Vector3(x=0.0, y=0.0, z=0.0)
        req.wrench.torque = Vector3(x=0.0, y=0.0, z=0.0)
        
        # Start immediately (Time(0)) and last for 2 seconds.
        req.start_time = rospy.Time.now()
        req.duration = rospy.Duration(1000)
        
        resp = apply_wrench(req)
        rospy.loginfo("Wrench applied successfully: %s", resp)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

if __name__ == '__main__':
    apply_force()
