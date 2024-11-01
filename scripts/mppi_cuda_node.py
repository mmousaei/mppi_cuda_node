#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import numpy as np
import rospy
import torch
import joblib
import time

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
# from std_msgs.msg import Float64
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from mavlink_transmitter import MavlinkTransmitter
from geometry_msgs.msg import Vector3Stamped
from gazebo_msgs.msg import LinkStates

# from mppi_numba import MPPI_Numba, Config
from lqr_controller import LqrController

from scipy.spatial.transform import Rotation

class ControlHexarotor:
    def __init__(self):
        rospy.init_node('hexarotor_controller', anonymous=True)
        
        self.initialize_hexarotor_parameters()
        self.frame_rate = 50
        self.cnt = 0
        self.start_time = 0
        self.rwheel_speed = 0
        
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug', Vector3Stamped, queue_size=10) #add new topic here
        self.rwheel_pub = rospy.Publisher('/mppi_debug/rwheel_data', Vector3Stamped, queue_size=10)
        # self.rwheel_control_pub = rospy.Publisher('/rwheel_effort', Float64, queue_size=10)
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.linkstate_callback)

        ### change rwheel control to torque control ####

        self.current_state = np.zeros(13)  # Placeholder for state from sensors
        self.control_inputs = WrenchStamped()
        self.odom = Odometry()
        

        self.states, self.actions, self.contact_forces, self.contact_torques, self.predicted_states, self.predicted_dynamics_agg, self.predicted_contact_forces, self.next_states, self.next_forces = [], [], [], [], [], [], [], [], []
        self.lqr_actions, self.mppi_dynamics_updates = [], []

        # self.cfg = Config(
        #     T=1.0,  # Horizon length in seconds
        #     dt=0.02,  # Time step
        #     num_control_rollouts=1024,  # Number of control sequences to sample
        #     num_controls=6,  # Dimensionality of control inputs
        #     num_states=12,  # Dimensionality of system states
        #     num_vis_state_rollouts=1,  # For visualization purposes
        #     seed=1
        # )
        # self.optimal_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        # self.mppi_controller = MPPI_Numba(self.cfg)
        # self.mppi_params = {
        #     'dt': self.cfg.dt,
        #     'x0': self.current_state,
        #     'xgoal': np.array([1, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #     'goal_tolerance': 0.001,
        #     'dist_weight': 2000,
        #     'lambda_weight': 20.0,
        #     'num_opt': 5,
        #     'u_std': np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]) * 0.05,
        #     'vrange': np.array([-10.0, 10.0]),
        #     'wrange': np.array([-1, 1]),
        # }
        # self.mppi_controller.set_params(self.mppi_params)
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()
        self.activate = False

        # Lqr parameters
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix
        self.lqr_controller.J_rw = self.rwheel_inertia_matrix

    def initialize_hexarotor_parameters(self):
        # Set hexarotor parameters like mass, inertia matrix, etc.
        rwheel_mass = 1.5
        self.hex_mass = 2.57 + rwheel_mass  # example value, replace with actual mass
        self.gravity_compensation = self.hex_mass * 9.81
        self.inertia_flat = np.array([0.115125971,  0.116524229,  0.230387752])
        self.rwheel_inertia = np.array([0.041, 0.041, 0.08185]) # Need to update all of these things with real values
        self.inertia_matrix = np.zeros((3, 3))
        np.fill_diagonal(self.inertia_matrix, self.inertia_flat)
        # Factor in inertia from the reaction wheel:
        self.inertia_matrix[0][0] += self.rwheel_inertia[2] # On the same axis
        self.inertia_matrix[1][1] += self.rwheel_inertia[0] + rwheel_mass * 0.5*0.5 # Mass shifted by 0.5 (|| axis thm)
        self.inertia_matrix[2][2] += self.rwheel_inertia[1] # On the same axis
        self.rwheel_inertia_matrix = np.zeros((3, 3))
        np.fill_diagonal(self.rwheel_inertia_matrix, self.rwheel_inertia)

    def activate_callback(self, data):
        self.activate = data.data
    def linkstate_callback(self, data):
        self.current_state[12] = -data.twist[19].angular.x
    def odometry_callback(self, data):
        # Update current state with odometry data
        self.odom = data
        pose = data.pose.pose
        twist = data.twist.twist
        #print("odom")
        self.current_state[:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.current_state[3:6] = [twist.linear.x, twist.linear.y, twist.linear.z]
        
        # Convert quaternion to euler angles
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        euler = euler_from_quaternion(quaternion)
        self.current_state[6:9] = euler
        # use transmitter to send control inputs
        self.current_state[9:12] = [twist.angular.x, twist.angular.y, twist.angular.z]
        att_msg = Vector3Stamped()
        att_msg.header.stamp = data.header.stamp
        att_msg.vector.x = euler[0]
        att_msg.vector.y = euler[1]
        att_msg.vector.z = euler[2]
        self.att_debug_pub.publish(att_msg)

    def target_callback(self, data):
        print("Target Received")
        self.mppi_params['xgoal'] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z, \
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
        quat = [0.0, -control_inputs[0], control_inputs[1], control_inputs[2]]
        # quat = [0.0, -0.0, 0.0, control_inputs[2]]
        # angular_rates = [-control_inputs[3], control_inputs[4], -control_inputs[5]]
        angular_rates = [control_inputs[3], -control_inputs[4], -control_inputs[5]]
        # angular_rates = [0, 0, 0]
        thrust = control_inputs[2]
        rwheel_acc = control_inputs[6]
        
        I = 0.08185 # wheel-dependent
        max_torque = 4.962 # motor-dependent
        max_acc = max_torque / I

        if self.cnt < 200:
            rwheel_acc = 0
        elif self.cnt % 200 < 50:
            rwheel_acc = max_acc


        # calculate rwheel_speed from the torque
        delta_t = time.time() - self.start_time
        print("Delta t =", delta_t)
        self.rwheel_speed = self.rwheel_speed + rwheel_acc * delta_t
        self.start_time = time.time()

        rwheel_speed_input = self.rwheel_speed / 148

        # use transmitter to send control inputs
        print(control_inputs)
        print(self.cnt)
        if self.activate:
            self.transmitter.send_attitude_control(angular_rates, thrust, quat, rwheel_speed_input)
            # effort = Float64()
            # effort.data = rwheel_acc  # Desired effort value
            # self.rwheel_control_pub.publish(effort)
            self.cnt += 1
        self.control_pub.publish(self.control_inputs)

        rwheel_msg = Vector3Stamped()
        rwheel_msg.header.stamp = rospy.Time.now()
        rwheel_msg.vector.x = rwheel_acc
        rwheel_msg.vector.y = self.rwheel_speed
        rwheel_msg.vector.z = self.current_state[12]
        self.rwheel_pub.publish(rwheel_msg)
        print(self.current_state)

    def compute_gravity_compensation(self):
        # print("compute gravity compensation")
        if self.odom.pose.pose.orientation.x == self.odom.pose.pose.orientation.y == self.odom.pose.pose.orientation.z == 0:
            return np.array([0, 0, self.hex_mass*9.81])
        orientation_quat = [self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, \
                            self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]
        #print(orientation_quat)
        gravity_vector_world = np.array([0, 0, self.hex_mass*9.81])
        rotation_matrix = Rotation.from_quat(orientation_quat).as_matrix()
        gravity_vector_body = rotation_matrix.dot(gravity_vector_world)
        return gravity_vector_body
    def normalize_control_inputs(self, control_inputs):
        # print("normalize control inputs")
        control_inputs[:2] = -control_inputs[0:2] / 40
        control_inputs[2] = -control_inputs[2] / (20 + self.hex_mass*9.81)
        control_inputs[3:6] = control_inputs[3:6] / 20
        return control_inputs
    def compute_control(self):
        # print("compute control")
        # print(f'current_state: {self.current_state[0]}, {self.current_state[1]}, {self.current_state[2]}')
        self.mppi_controller.shift_and_update(self.current_state, self.optimal_control_seq, num_shifts=1)
        self.optimal_control_seq = self.mppi_controller.solve()
        control_inputs = self.optimal_control_seq[0, :]  # Use the first set of control inputs from the optimized sequence
        gravity_vector_body = self.compute_gravity_compensation()
        control_inputs[:3] += gravity_vector_body
        control_inputs = self.normalize_control_inputs(control_inputs)
        self.publish_cmd(control_inputs)

    def compute_control_lqr(self):
        # print("compute control lqr")
        control_inputs = self.lqr_controller.lqr_control(self.current_state.copy())
        gravity_vector_body = self.compute_gravity_compensation()
        control_inputs[:3] += gravity_vector_body
        control_inputs = self.normalize_control_inputs(control_inputs)
        self.publish_cmd(control_inputs)
        

    

    def spin(self):
        print("spin")
        rate = rospy.Rate(self.frame_rate)
        while not rospy.is_shutdown():
            # self.compute_control()
            #print(self.current_state)
            self.compute_control_lqr()
            rate.sleep()

def main():
    controller = ControlHexarotor()
    controller.spin()

if __name__ == "__main__":
    main()
