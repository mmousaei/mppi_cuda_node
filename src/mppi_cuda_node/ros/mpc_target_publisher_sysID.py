#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from tf.transformations import quaternion_from_euler

def send_excitation_trajectory(start, max_duration=10, pub_rate=5, excitation_amp=0.1, excitation_freq=0.1, excite_roll=True, excite_pitch=True, excite_yaw=True):
    pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=10)
    rate = rospy.Rate(pub_rate)
    num_steps = int(max_duration * pub_rate)
    
    rospy.loginfo("Publishing {} waypoints with attitude excitation.".format(num_steps))
    
    for i in range(num_steps):
        time = i / float(pub_rate)  # Time for sinusoidal excitation
        roll = excitation_amp * np.sin(2 * np.pi * excitation_freq * time) if excite_roll else 0.0
        pitch = excitation_amp * np.sin(2 * np.pi * excitation_freq * time) if excite_pitch else 0.0
        yaw = excitation_amp * np.sin(2 * np.pi * excitation_freq * time) if excite_yaw else 0.0
        
        quat = quaternion_from_euler(roll, pitch, yaw)
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = start[0]
        pose_msg.pose.position.y = start[1]
        pose_msg.pose.position.z = start[2]
        pose_msg.pose.orientation.x = roll
        pose_msg.pose.orientation.y = pitch
        pose_msg.pose.orientation.z = yaw
        pose_msg.pose.orientation.w = 1
        
        pub.publish(pose_msg)
        rospy.loginfo("Published waypoint {}: Position {} Attitude [roll: {}, pitch: {}, yaw: {}]".format(i, start, roll, pitch, yaw))
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('excitation_target_publisher')
    start_position = [0.5, 0.5, 1.5]
    try:
        send_excitation_trajectory(start_position)
    except rospy.ROSInterruptException:
        pass
