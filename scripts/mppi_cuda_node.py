#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import numpy as np
import rospy
import torch
import joblib

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from mavlink_transmitter import MavlinkTransmitter
from geometry_msgs.msg import Vector3Stamped

# from mppi_numba_retune import MPPI_Numba, Config
# from mppi_numba import MPPI_Numba, Config
# from mppi_numba_fast import MPPI_Numba, Config
from mppi_numba_gravity import MPPI_Numba, Config
from lqr_controller import LqrController
# from lqr_controller_gravity import LqrController

from scipy.spatial.transform import Rotation

class ControlHexarotor:
    def __init__(self):
        rospy.init_node('hexarotor_controller', anonymous=True)
        self.initialize_hexarotor_parameters()
        self.frame_rate = 50
        self.cnt = 0
        
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug', Vector3Stamped, queue_size=10)
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)

        self.current_state = np.zeros(12)  # Placeholder for state from sensors
        self.control_inputs = WrenchStamped()
        self.odom = Odometry()
        
        
        self.states, self.actions, self.contact_forces, self.contact_torques, self.predicted_states, self.predicted_dynamics_agg, self.predicted_contact_forces, self.next_states, self.next_forces = [], [], [], [], [], [], [], [], []
        self.lqr_actions, self.mppi_dynamics_updates = [], []

        self.cfg = Config(
            T=0.5,  # Horizon length in seconds
            dt=0.02,  # Time step
            num_control_rollouts=1024,  # Number of control sequences to sample
            num_controls=6,  # Dimensionality of control inputs
            num_states=12,  # Dimensionality of system states
            num_vis_state_rollouts=1,  # For visualization purposes
            seed=1
        )
        self.optimal_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        self.mppi_controller = MPPI_Numba(self.cfg)
        self.mppi_params = {
            'dt': self.cfg.dt,
            'x0': self.current_state,
            'xgoal': np.array([1, -1, 2, 0, 0, 0, -0.0, 0.0, -0.0, 0, 0, 0]),
            'goal_tolerance': 0.001,
            'dist_weight': 2000,
            'lambda_weight': 1.0,
            'num_opt': 2,
            # 'u_std': np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])*0.05,
            'u_std': np.array([0.2, 0.2, 0.5, 0.005, 0.005, 0.003])*3,
            'vrange': np.array([-80.0, 80.0]),
            'wrange': np.array([-0.5, 0.5]),
            'weights' : np.array([4550, 4550, 5300, 150, 75000, 35000, 85000, 500, 1, 5, 5, 1, 500]), # w_pose_x, w_pose_y, w_pose_z, w_vel, w_att_roll, w_att_pitch, w_att_yaw, w_omega, w_cont, w_cont_m, w_cont_f, w_cont_M, w_terminal
            # 'weights' : np.array([x, y, z, v, r, p, y, w, f1, f2, m2, m1, t]), # w_pose_x, w_pose_y, w_pose_z, w_vel, w_att_roll, w_att_pitch, w_att_yaw, w_omega, w_cont, w_cont_m, w_cont_f, w_cont_M, w_terminal
            'inertia_mass' : np.array([self.inertia_flat[0], self.inertia_flat[1], self.inertia_flat[2], self.hex_mass]) # I_xx, I_yy, I_zz, mass
        }
        
        self.mppi_controller.set_params(self.mppi_params)
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()
        self.activate = False

        # Lqr parameters
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        
        self.lqr_controller.J = self.inertia_matrix
        self.lqr_controller.desired_x = self.mppi_params['xgoal']


    def initialize_hexarotor_parameters(self):
        # Set hexarotor parameters like mass, inertia matrix, etc.
        # self.hex_mass = 2.57  # example value, replace with actual mass
        self.hex_mass = 3.49 # 2.68(base_link)+0.1(cgo3_mount_link)+0.1(cgo3_vertical_arm_link)+0.1(cgo3_horizontal_arm_link)+0.1(cgo3_camera_link)+0.1(sim_ft_sensor)+0.1(left_leg)+0.1(right_leg)+0.015(imu_link)+0.03(6*rotors[each 0.005])=3.425kg
        # self.hex_mass = 1 # 2.68(base_link)+0.1(cgo3_mount_link)+0.1(cgo3_vertical_arm_link)+0.1(cgo3_horizontal_arm_link)+0.1(cgo3_camera_link)+0.1(sim_ft_sensor)+0.1(left_leg)+0.1(right_leg)+0.015(imu_link)+0.03(6*rotors[each 0.005])=3.425kg
        self.gravity_compensation_scale = 1
        # self.inertia_flat = np.array([0.115125971,  0.116524229,  0.230387752])
        self.inertia_flat = np.array([0.05132,  0.06282,  0.06248]) #  ​                     0.05132 0.00000 −0.00232
                                                                    # Full inertia matrix:  ​0.00000 0.06282  0.00000
                                                                    # ​                     −0.00232 0.00000  0.06248​
        self.inertia_matrix = np.zeros((3, 3))
        np.fill_diagonal(self.inertia_matrix, self.inertia_flat)

    def activate_callback(self, data):
        self.activate = data.data
    def odometry_callback(self, data):
        # Update current state with odometry data
        self.odom = data
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
        att_msg = Vector3Stamped()
        att_msg.header.stamp = data.header.stamp
        att_msg.vector.x = euler[0]
        att_msg.vector.y = euler[1]
        att_msg.vector.z = euler[2]
        self.att_debug_pub.publish(att_msg)

    def target_callback(self, data):
        print("Target Received")
        
        self.mppi_controller.params['xgoal'] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z, \
                                              0, 0, 0, \
                                              data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, \
                                              0, 0, 0])
        self.lqr_controller.desired_x = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z, \
                                              0, 0, 0, \
                                              data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, \
                                              0, 0, 0])
    def publish_cmd(self, control_inputs):
        self.control_inputs.header.stamp = rospy.Time.now()
        self.control_inputs.wrench.force.x = control_inputs[0]
        self.control_inputs.wrench.force.y = control_inputs[1]
        self.control_inputs.wrench.force.z = control_inputs[2]
        self.control_inputs.wrench.torque.x = control_inputs[3]
        self.control_inputs.wrench.torque.y = control_inputs[4]
        self.control_inputs.wrench.torque.z = control_inputs[5]
        # quat = [0.0, control_inputs[0], control_inputs[1], control_inputs[2]]
        quat = [0.0, control_inputs[0], -control_inputs[1], -control_inputs[2]]
        # quat = [0.0, -0.0, 0.0, control_inputs[2]]
        # angular_rates = [-control_inputs[3], control_inputs[4], -control_inputs[5]]
        angular_rates = [control_inputs[3], -control_inputs[4], -control_inputs[5]]
        # angular_rates = [0, 0, 0]
        thrust = -control_inputs[2]
        # use transmitter to send control inputs
        if self.activate:
            self.transmitter.send_attitude_control(angular_rates, thrust, quat)
        self.control_pub.publish(self.control_inputs)

    def compute_gravity_compensation(self):
        # print("compute gravity compensation")
        orientation_quat = [self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, \
                            self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]
        orientation_quat = [0, 0, 0, 1]
        gravity_vector_world = np.array([0, 0, -self.hex_mass*9.81])
        rotation_matrix = Rotation.from_quat(orientation_quat).as_matrix()
        gravity_vector_body = rotation_matrix.T.dot(gravity_vector_world)
        return gravity_vector_body
    def normalize_control_inputs(self, control_inputs, gravity_vector_body):
        # print("normalize control inputs")
        control_inputs[0] = control_inputs[0] / (29.64)
        control_inputs[1] = control_inputs[1] / (26.96)
        control_inputs[2] = control_inputs[2] / (61.78)
        control_inputs[3:6] = control_inputs[3:6] * 2
        return control_inputs
    def normalize_control_inputs_lqr(self, control_inputs, gravity_vector_body):
        # print("normalize control inputs")
        # Adjust normalization for the z-direction based on the tilt (roll and pitch)
        
        tilt_factor = np.cos(self.current_state[6]) * np.cos(self.current_state[7])
        # control_inputs[0] = control_inputs[0] / (29.64 - 0.35*np.abs(gravity_vector_body[0]))
        # control_inputs[1] = control_inputs[1] / (26.96 + 0.35*np.abs(gravity_vector_body[1])) * self.gravity_compensation_scale
        # control_inputs[2] = control_inputs[2] / (181.266781 - 3.5*gravity_vector_body[2]) * self.gravity_compensation_scale
        control_inputs[0] = control_inputs[0] / (29.64)
        control_inputs[1] = control_inputs[1] / (26.96) 
        control_inputs[2] = control_inputs[2] / (61.78) 
        
        # control_inputs[2] = control_inputs[2] / (61.78)
        control_inputs[3:6] = control_inputs[3:6] / 10
        return control_inputs
    def dynamics_update(self, current_state, control_inputs, rate):
        k1 = self.lqr_controller.hex_dynamics(current_state, control_inputs) * rate
        k2 = self.lqr_controller.hex_dynamics(current_state + k1 / 2, control_inputs) * rate
        k3 = self.lqr_controller.hex_dynamics(current_state + k2 / 2, control_inputs) * rate
        k4 = self.lqr_controller.hex_dynamics(current_state + k3, control_inputs) * rate
        next_state = current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state
    def compute_control(self):
        # print(f'current_state: {self.current_state[0]}, {self.current_state[1]}, {self.current_state[2]}')
        self.mppi_controller.shift_and_update(self.current_state, self.optimal_control_seq, num_shifts=1)
        self.optimal_control_seq = self.mppi_controller.solve()
        control_inputs = self.optimal_control_seq[0, :]  # Use the first set of control inputs from the optimized sequence
        gravity_vector_body = self.compute_gravity_compensation()
        
        # control_inputs[:3] -= gravity_vector_body
        # control_inputs[2] += gravity_vector_body
        control_inputs = self.normalize_control_inputs(control_inputs, gravity_vector_body)
        self.publish_cmd(control_inputs)
    
    def compute_control_force_angular_rate(self):
        # print("compute control")
        # print(f'current_state: {self.current_state[0]}, {self.current_state[1]}, {self.current_state[2]}')
        self.mppi_controller.shift_and_update(self.current_state, self.optimal_control_seq, num_shifts=1)
        self.optimal_control_seq = self.mppi_controller.solve()
        control_inputs = self.optimal_control_seq[0, :]  # Use the first set of control inputs from the optimized sequence
        gravity_vector_body = self.compute_gravity_compensation()
        control_inputs[:3] -= gravity_vector_body
        next_state = self.dynamics_update(self.current_state, control_inputs, 0.02) 
        control_inputs = self.normalize_control_inputs(control_inputs, gravity_vector_body)
        rate_controls = np.array([control_inputs[0], control_inputs[1], control_inputs[2], next_state[9], next_state[10], next_state[11]])
        # rate_controls = self.normalize_control_inputs(rate_controls)
        self.publish_cmd(rate_controls)
        # self.publish_cmd(control_inputs)

    def compute_control_lqr(self):
        # print("compute control lqr")
        control_inputs = self.lqr_controller.lqr_control(self.current_state.copy())
        gravity_vector_body = self.compute_gravity_compensation()
        print(f"gravity_vector_body: {gravity_vector_body}")
        # gravity_vector_body[0] = 0
        # gravity_vector_body[1] = 0
        print(f"command_before_gravity: {control_inputs[:3]}")
        control_inputs[:3] -= (gravity_vector_body)
        next_state = self.dynamics_update(self.current_state, control_inputs, 0.02)
        control_inputs = self.normalize_control_inputs_lqr(control_inputs, gravity_vector_body)
        rate_controls = np.array([control_inputs[0], control_inputs[1], control_inputs[2], next_state[9], next_state[10], -next_state[11]])
        # rate_controls = self.normalize_control_inputs_lqr(control_inputs, gravity_vector_body)
        # rate_controls[:3] += (gravity_vector_body / 45.45)
        
        # control_inputs[0] = -0.5
        # control_inputs[1] = 0
        # control_inputs[2] += 0.1
        # control_inputs[3] = -0.1
        # control_inputs[5] = -0.1
        self.publish_cmd(control_inputs)
        

    

    def spin(self):
        print("spin")
        rate = rospy.Rate(self.frame_rate)
        while not rospy.is_shutdown():
            self.compute_control()
            # self.compute_control_force_angular_rate()
            # self.compute_control_lqr()
            rate.sleep()

def main():
    controller = ControlHexarotor()
    controller.spin()


if __name__ == "__main__":
    main()
