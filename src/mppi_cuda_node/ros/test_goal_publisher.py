#!/usr/bin/env python3
"""
Test Goal Publisher for MPPI ROS Wrapper
Publishes dynamic goals to test the MPPI ROS wrapper architecture
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import tf.transformations as tf_trans

class TestGoalPublisher:
    """Publishes test goals, odometry, and IMU data for MPPI testing"""
    
    def __init__(self):
        rospy.init_node('test_goal_publisher', anonymous=True)
        
        # Publishers
        self.goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu', Imu, queue_size=1)
        
        # State variables
        self.test_counter = 0
        self.current_pos = np.array([0.0, 0.0, -1.0])
        self.current_vel = np.array([0.0, 0.0, 0.0])
        self.current_att = np.array([0.0, 0.0, 0.0])  # Roll, pitch, yaw
        
        rospy.loginfo("Test Goal Publisher initialized")
    
    def publish_goal(self):
        """Publish dynamic goal"""
        time_factor = self.test_counter * 0.05
        
        # Circular motion goal
        goal_x = 0.5 * np.sin(time_factor)
        goal_y = 0.5 * np.cos(time_factor)
        goal_z = -1.0 + 0.2 * np.sin(time_factor * 0.5)  # Varying height
        
        # Create goal message
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "world"
        
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.position.z = goal_z
        
        # Set orientation to identity quaternion
        goal_msg.pose.orientation.w = 1.0
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        
        self.goal_pub.publish(goal_msg)
        
        return goal_x, goal_y, goal_z
    
    def publish_odometry(self):
        """Publish simulated odometry"""
        # Simple state evolution
        dt = 0.02
        
        # Add some movement
        self.current_pos[0] += dt * 0.1 * np.sin(self.test_counter * 0.1)
        self.current_pos[1] += dt * 0.1 * np.cos(self.test_counter * 0.1)
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "base_link"
        
        # Position
        odom_msg.pose.pose.position.x = self.current_pos[0]
        odom_msg.pose.pose.position.y = self.current_pos[1]
        odom_msg.pose.pose.position.z = self.current_pos[2]
        
        # Orientation (quaternion from euler)
        quat = tf_trans.quaternion_from_euler(self.current_att[0], self.current_att[1], self.current_att[2])
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        
        # Velocity
        odom_msg.twist.twist.linear.x = self.current_vel[0]
        odom_msg.twist.twist.linear.y = self.current_vel[1]
        odom_msg.twist.twist.linear.z = self.current_vel[2]
        
        self.odom_pub.publish(odom_msg)
    
    def publish_imu(self):
        """Publish simulated IMU data"""
        imu_msg = Imu()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "base_link"
        
        # Orientation (quaternion from euler)
        quat = tf_trans.quaternion_from_euler(self.current_att[0], self.current_att[1], self.current_att[2])
        imu_msg.orientation.x = quat[0]
        imu_msg.orientation.y = quat[1]
        imu_msg.orientation.z = quat[2]
        imu_msg.orientation.w = quat[3]
        
        # Angular velocity (set to zero for simplicity)
        imu_msg.angular_velocity.x = 0.0
        imu_msg.angular_velocity.y = 0.0
        imu_msg.angular_velocity.z = 0.0
        
        # Linear acceleration (set to gravity)
        imu_msg.linear_acceleration.x = 0.0
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = -9.81
        
        self.imu_pub.publish(imu_msg)
    
    def run_test(self):
        """Run test loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown() and self.test_counter < 100:
            try:
                # Publish goal
                goal_x, goal_y, goal_z = self.publish_goal()
                
                # Publish odometry
                self.publish_odometry()
                
                # Publish IMU
                self.publish_imu()
                
                # Log every 10th iteration
                if self.test_counter % 10 == 0:
                    rospy.loginfo(f"Publishing: Goal=[{goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f}], "
                                f"State=[{self.current_pos[0]:.2f}, {self.current_pos[1]:.2f}, {self.current_pos[2]:.2f}]")
                
                self.test_counter += 1
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in test loop: {e}")
                rate.sleep()
        
        rospy.loginfo("Test completed")

if __name__ == '__main__':
    try:
        publisher = TestGoalPublisher()
        publisher.run_test()
    except Exception as e:
        rospy.logerr(f"Failed to start Test Goal Publisher: {e}")
