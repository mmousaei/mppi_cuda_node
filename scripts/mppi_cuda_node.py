#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import numpy as np
import rospy
import torch
import joblib
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mppi_numba import MPPI_Numba, Config
from tf.transformations import euler_from_quaternion
from mavlink_transmitter import MavlinkTransmitter

class ControlHexarotor:
    def __init__(self):
        rospy.init_node('hexarotor_controller', anonymous=True)
        
        self.initialize_hexarotor_parameters()
        self.frame_rate = 60
        self.cnt = 0
        
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)

        self.current_state = np.zeros(12)  # Placeholder for state from sensors
        self.control_inputs = Float32MultiArray()

        self.states, self.actions, self.contact_forces, self.contact_torques, self.predicted_states, self.predicted_dynamics_agg, self.predicted_contact_forces, self.next_states, self.next_forces = [], [], [], [], [], [], [], [], []
        self.lqr_actions, self.mppi_dynamics_updates = [], []

        self.cfg = Config(
            T=1.0,  # Horizon length in seconds
            dt=0.02,  # Time step
            num_control_rollouts=2048*2,  # Number of control sequences to sample
            num_controls=6,  # Dimensionality of control inputs
            num_states=12,  # Dimensionality of system states
            num_vis_state_rollouts=1,  # For visualization purposes
            seed=1
        )
        self.mppi_controller = MPPI_Numba(self.cfg)
        self.mppi_params = {
            'dt': self.cfg.dt,
            'x0': self.current_state,
            'xgoal': np.array([1, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'goal_tolerance': 0.001,
            'dist_weight': 2000,
            'lambda_weight': 20.0,
            'num_opt': 5,
            'u_std': np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]) * 1,
            'vrange': np.array([-10.0, 10.0]),
            'wrange': np.array([-1, 1]),
        }
        self.mppi_controller.set_params(self.mppi_params)
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()
        self.activate = False

    def initialize_hexarotor_parameters(self):
        # Set hexarotor parameters like mass, inertia matrix, etc.
        self.hex_mass = 2  # example value, replace with actual mass
        self.gravity_compensation = self.hex_mass * 9.81
        self.inertia_flat = np.array([0.115125971,  0.116524229,  0.230387752])
        self.inertia_matrix = np.zeros((3, 3))
        np.fill_diagonal(self.inertia_matrix, self.inertia_flat)

    def odometry_callback(self, data):
        # Update current state with odometry data
        pose = data.pose.pose
        twist = data.twist.twist
        # print("odom")
        self.current_state[:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.current_state[3:6] = [twist.linear.x, twist.linear.y, twist.linear.z]
        
        # Convert quaternion to euler angles
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        euler = euler_from_quaternion(quaternion)
        self.current_state[6:9] = euler
        # use transmitter to send control inputs
        self.current_state[9:] = [twist.angular.x, twist.angular.y, twist.angular.z]

    def target_callback(self, data):
        # print("Target Received")
        self.mppi_params['xgoal'] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z, \
                                              data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, \
                                              data.twist.linear.x, data.twist.linear.y, data.twist.linear.z, \
                                              data.twist.angular.x, data.twist.angular.y, data.twist.angular.z])

    def compute_control(self):
        # print("compute control")
        optimal_control_seq = self.mppi_controller.solve()
        control_inputs = optimal_control_seq[0, :]  # Use the first set of control inputs from the optimized sequence
        self.control_inputs.data = control_inputs
        quat = [0, control_inputs[0], control_inputs[1], control_inputs[2]]
        angular_rates = [control_inputs[3], control_inputs[4], control_inputs]
        thrust = control_inputs[2]
        # use transmitter to send control inputs
        if self.activate == True:
            self.transmitter.send_attitude_control(angular_rates, thrust, quat)
        self.control_pub.publish(self.control_inputs)

    def spin(self):
        print("spin")
        rate = rospy.Rate(self.frame_rate)
        while not rospy.is_shutdown():
            self.compute_control()
            rate.sleep()

def main():
    controller = ControlHexarotor()
    controller.spin()

if __name__ == "__main__":
    main()
