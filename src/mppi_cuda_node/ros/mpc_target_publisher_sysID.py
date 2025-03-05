#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from tf.transformations import quaternion_from_euler

def send_excitation_trajectory(start, max_duration=10, pub_rate=5, excitation_amp=0.01, excitation_freq=0.05, excite_roll=True, excite_pitch=False, excite_yaw=False):
    pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=10)
    rate = rospy.Rate(pub_rate)
    num_steps = int(max_duration * pub_rate)
    total_time = num_steps / float(pub_rate)
    complete_cycles = np.ceil(total_time * excitation_freq)  # Ensure we finish full cycles
    adjusted_max_duration = complete_cycles / excitation_freq
    num_steps = int(adjusted_max_duration * pub_rate)  # Recalculate steps
    
    rospy.loginfo("Publishing {} waypoints with attitude excitation, ensuring full cycle completion.".format(num_steps))
    
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
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        pub.publish(pose_msg)
        rospy.loginfo("Published waypoint {}: Position {} Attitude [roll: {}, pitch: {}, yaw: {}]".format(i, start, roll, pitch, yaw))
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('excitation_target_publisher')
    start_position = [0.0, 0.0, 0.8]
    try:
        send_excitation_trajectory(start_position)
    except rospy.ROSInterruptException:
        pass