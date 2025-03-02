#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np

def send_interpolated_targets(start, target, max_step=0.01, pub_rate=5):
    # Calculate the Euclidean distance between start and target
    distance = np.linalg.norm(np.array(target) - np.array(start))
    # Determine the number of waypoints (segments)
    num_steps = int(np.ceil(distance / max_step))
    
    pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=10)
    rate = rospy.Rate(pub_rate)
    
    rospy.loginfo("Publishing {} waypoints.".format(num_steps))
    
    for i in range(1, num_steps + 1):
        # Calculate the interpolation factor
        alpha = float(i) / num_steps
        # Compute the interpolated position
        pos = np.array(start) + alpha * (np.array(target) - np.array(start))
        
        # Create and populate the PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"  # Adjust the frame if needed
        
        pose_msg.pose.position.x = pos[0]
        pose_msg.pose.position.y = pos[1]
        pose_msg.pose.position.z = pos[2]
        
        # Set orientation; adjust as necessary. (Make sure the quaternion is normalized.)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        # Publish the message
        pub.publish(pose_msg)
        rospy.loginfo("Published waypoint {}: {}".format(i, pos))
        
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('interpolated_target_publisher')
    # Example start and target positions [x, y, z]
    start_position = [0.5, 0.5, 1.5]
    target_position = [0.0, 0.0, 0.8]  # Change as needed
    
    try:
        send_interpolated_targets(start_position, target_position)
    except rospy.ROSInterruptException:
        pass
