#!/usr/bin/env python
"""
This node runs the MPC controller:
  - It subscribes to odometry and the target published by the MPPI node (on '/mppi/target').
  - It computes a control command using one-step MPC (or tube MPC).
  - It publishes control commands and (if activated) sends them via MAVLink.
"""

import os
import sys
import numpy as np
import math
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped, PoseStamped, Vector3Stamped
from core_trajectory_msgs.msg import FixedTrajectory
from diagnostic_msgs.msg import KeyValue
from tf.transformations import euler_from_quaternion

# --- MPC imports ---
from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC
from mppi_cuda_node.controllers.mpc.acados.acados_mpc_tube import TubeMPC
from mppi_cuda_node.controllers.lqr.lqr_controller import LqrController
from mppi_cuda_node.misc.mavlink.mavlink_transmitter import MavlinkTransmitter
from scipy.spatial.transform import Rotation

from dynamic_reconfigure.server import Server
# from mppi_cuda_node.cfg.MPCParamsConfig import MPCParamsConfig
import mppi_cuda_node.cfg.MPCParamsConfig as MPCParamsConfig



class MPCControllerNode(object):
    def __init__(self):
        rospy.init_node('mpc_controller', anonymous=True)
        rospy.loginfo("Initializing MPC Controller Node ...")

        self.current_state = np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.mpc_target = np.zeros(12)     # To be updated from the MPPI node
        self.mpc_target[2] = 0.8
        self.activate = False

        self.initialize_hexarotor_parameters()

        # ----- MPC Setup -----
        self.mpc_params = {
            'inertia': self.inertia_flat,
            'mass': self.hex_mass,
            'horizon': 30,
            'gravity': 9.81,
            'max_force': 10.0,
            'max_torque': 1.0,
            'control_weight': 0.4,
            'tracking_weight_pos': 100,
            'tracking_weight_vel': 3,
            'tracking_weight_att': 100,
            'tracking_weight_ang_vel': 4,
            'terminal_weight': 0.1,
            'smoothness_weight': 0.05,
            'dt': 0.01
        }
        self.mpc_tube_params = {
            'inertia': self.inertia_flat,
            'mass': self.hex_mass,
            'horizon': 5,
            'gravity': 9.81,
            'max_force': 10.0,
            'max_torque': 1,
            'control_weight': 0.2,
            'tracking_weight_pos': 24,
            'tracking_weight_vel': 3,
            'tracking_weight_att': 10,
            'tracking_weight_ang_vel': 0.1,
            'smoothness_weight': 0.05,
            'lqr_weights': np.array([1, 0.2, 0.05, 0.01, 2e-3, 2e-3]),
            'dt': 0.2
        }
        self.mpc = MPC(self.mpc_params)
        self.tube_mpc = TubeMPC(self.mpc_tube_params)

        # LQR controller for auxiliary purposes (if needed)
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix

        # ----- Publishers -----
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug', Vector3Stamped, queue_size=10)
        self.fixed_traj_pub = rospy.Publisher("/fixed_trajectory", FixedTrajectory, queue_size=10)
        self.ft_pub = rospy.Publisher("/ft_filtered", WrenchStamped, queue_size=10)

        # ----- MAVLink transmitter -----
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()

        # ----- Subscribers -----
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mpc/target', PoseStamped, self.mpc_target_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)
        rospy.Subscriber('/ft_data', WrenchStamped, self.force_sensor_callback)

        self.mpc_rate_hz = 100.0  # Run MPC at 50 Hz
        self.last_time_pid_pos_publish = rospy.Time.now()

        # Force sensor variable: we'll update this with sensor data.
        # Smoothing factor between 0 (very smooth) and 1 (no filtering)
        self.alpha = 0.05  
        # Initialize the filtered force vector (x, y, z)
        self.filtered_force = [0.0, 0.0, 0.0]
        self.initialized_filter = False
        # You might also want to keep the latest raw measurement if needed
        self.measured_force = [0.0, 0.0, 0.0]

        # Set up dynamic reconfigure server for tuning MPC parameters
        # self.dyn_server = Server(MPCParamsConfig, self.dynamic_reconfigure_callback)

        rospy.loginfo("MPC Controller Node Initialization Complete.")

    def initialize_hexarotor_parameters(self):
        # Set your hexarotor parameters (tweak as needed)
        self.hex_mass = 6.15  # kg (example value)
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        self.inertia_matrix = np.diag(self.inertia_flat)

    def dynamic_reconfigure_callback(self, config, level):
        rospy.loginfo("Reconfigure Request:\nhorizon = %d\ndt = %.3f\nmax_force = %.2f\nmax_torque = %.2f\ncontrol_weight = %.2f\ntracking_weight_pos = %.2f\ntracking_weight_vel = %.2f\ntracking_weight_att = %.2f\ntracking_weight_ang_vel = %.2f\nsmoothness_weight = %.2f",
                      config['horizon'],config['dt'],config['max_force'],config['max_torque'],config['control_weight'],config['tracking_weight_pos'],config['tracking_weight_vel'],config['tracking_weight_att'],config['tracking_weight_ang_vel'],config['smoothness_weight'])
        # Update your MPC parameters here
        self.mpc_params['horizon'] = config['horizon']
        self.mpc_params['dt'] = config['dt']
        self.mpc_params['max_force'] = config['max_force']
        self.mpc_params['max_torque'] = config['max_torque']
        self.mpc_params['control_weight'] = config['control_weight']
        self.mpc_params['tracking_weight_pos'] = config['tracking_weight_pos']
        self.mpc_params['tracking_weight_vel'] = config['tracking_weight_vel']
        self.mpc_params['tracking_weight_att'] = config['tracking_weight_att']
        self.mpc_params['tracking_weight_ang_vel'] = config['tracking_weight_ang_vel']
        self.mpc_params['smoothness_weight'] = config['smoothness_weight']
        self.mpc.update_parameters(self.mpc_params)
        return config

    def force_sensor_callback(self, msg):
        """
        Callback for force sensor data.
        Applies an exponential moving average filter to reduce high-frequency noise.
        """
        # Extract raw force data from the message
        raw_force = [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ]
        # On the first callback, initialize the filtered value with the raw data
        if not self.initialized_filter:
            self.filtered_force = raw_force
            self.initialized_filter = True
        else:
            # Apply the exponential moving average filter
            self.filtered_force = [
                self.alpha * raw + (1 - self.alpha) * filt
                for raw, filt in zip(raw_force, self.filtered_force)
            ]
        filtered_force_msg = WrenchStamped()

        filtered_force_msg = msg

        filtered_force_msg.wrench.force.x = self.filtered_force[0]
        filtered_force_msg.wrench.force.y = self.filtered_force[1]
        filtered_force_msg.wrench.force.z = self.filtered_force[2]

        self.ft_pub.publish(filtered_force_msg)
        
        # Update measured_force with the filtered value
        self.measured_force = self.filtered_force

    def activate_callback(self, data):
        self.activate = data.data

    def odometry_callback(self, data):
        self.odom = data
        pose = data.pose.pose
        twist = data.twist.twist

        # Update state: positions and velocities
        self.current_state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.current_state[3:6] = [twist.linear.x, twist.linear.y, twist.linear.z]

        # Convert quaternion to Euler angles
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        euler = euler_from_quaternion(quaternion)
        self.current_state[6:9] = euler

        self.current_state[6:9] += np.random.normal(loc=0.0, scale=0.01, size=self.current_state[6:9].shape)

        # Angular velocities
        self.current_state[9:] = [twist.angular.x, twist.angular.y, twist.angular.z]

        # Publish attitude debug message
        att_msg = Vector3Stamped()
        att_msg.header.stamp = data.header.stamp
        att_msg.vector.x = euler[0] #* 180 / np.pi
        att_msg.vector.y = euler[1] #* 180 / np.pi
        att_msg.vector.z = euler[2] #* 180 / np.pi
        self.att_debug_pub.publish(att_msg)

    def mpc_target_callback(self, data):
        """
        Callback for the target published by the MPPI node.
        We update our internal MPC target state accordingly.
        """
        self.mpc_target[0] = data.pose.position.x
        self.mpc_target[1] = data.pose.position.y
        self.mpc_target[2] = data.pose.position.z
        # self.mpc_target[0] = 0
        # self.mpc_target[1] = 0
        # self.mpc_target[2] = 0.8

        # For simplicity, we zero the remaining state elements.

        self.mpc_target[3:] = 0.0
        # self.mpc_target[6] = data.pose.orientation.x
        # self.mpc_target[7] = data.pose.orientation.y
        # self.mpc_target[8] = data.pose.orientation.z
        
    def normalize_control_inputs_mpc(self, ctrl):
        """
        Normalize/scaling for MPC outputs (body rates and thrust).
        Adjust these gains to suit your vehicle.
        """
        hover_thrust = 0.61
        ctrl[0] = ctrl[0] * 0.515336334
        ctrl[1] = ctrl[1] * 0.515336334
        ctrl[2] = ctrl[2] * hover_thrust / (self.hex_mass * 9.81)
        ctrl[3:6] = ctrl[3:6] * 0.5
        return ctrl

    def run_mpc(self):
        """
        Solve the one-step MPC using:
         - current_state
         - mpc_target (set by MPPI)
         - MPPI's first-control as the initial guess
        """
        # We'll use the first MPPI control as an initial guess
        # ctrl_guess_mppi = self.optimal_control_seq[0, :].copy()

        # For simplicity, just pass the raw guess in:
        u_mpc = self.mpc.compute_control(
            self.current_state.copy(),
            self.mpc_target.copy(),
            np.zeros(6),
            self.mpc_params['dt']
        )

        F_measured = self.measured_force
        F_des = np
        return u_mpc
    def run_mpc_tube(self):
        """
        Compute the control input using tube MPC.
        Uses the current state, the MPC target (from MPPI), and a nominal hover input.
        """
        x_real = self.current_state.copy()
        x_nom  = self.mpc_target.copy()

        # Nominal hover input: Fz = m*g, others = 0
        u_hover = np.array([0.0, 0.0, self.hex_mass * 9.81, 0.0, 0.0, 0.0])
        # Compute tube MPC control (or you can call self.mpc.compute_control if preferred)
        u_tube = self.tube_mpc.compute_control(x_real, x_nom, u_hover)
        return u_tube

    def publish_cmd(self, control_inputs):
        cmd_msg = WrenchStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.wrench.force.x = control_inputs[0]
        cmd_msg.wrench.force.y = control_inputs[1]
        cmd_msg.wrench.force.z = control_inputs[2]
        cmd_msg.wrench.torque.x = control_inputs[3]
        cmd_msg.wrench.torque.y = control_inputs[4]
        cmd_msg.wrench.torque.z = control_inputs[5]

        # Prepare attitude command parameters for the transmitter
        quat = [0.0, control_inputs[0], -control_inputs[1], -control_inputs[2]]
        angular_rates = [control_inputs[3], -control_inputs[4], -control_inputs[5]]
        thrust = -control_inputs[2]

        if self.activate:
            self.transmitter.send_attitude_control(angular_rates, thrust, quat)
            elapsed_time_pid = rospy.Time.now() - self.last_time_pid_pos_publish
            if elapsed_time_pid.to_sec() > 0.5:
                self.publish_position_pid()

        self.control_pub.publish(cmd_msg)

    def publish_position_pid(self):
        """
        Example function to publish a trajectory for debugging.
        """
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        z = self.odom.pose.pose.position.z

        traj = FixedTrajectory()
        traj.type = "Point"
        att1 = KeyValue(key="frame_id", value="world")
        att2 = KeyValue(key="height", value=str(z))
        att3 = KeyValue(key="max_acceleration", value=str(0.4))
        att4 = KeyValue(key="velocity", value=str(0.1))
        att5 = KeyValue(key="x", value=str(x))
        att6 = KeyValue(key="y", value=str(y))
        traj.attributes.extend([att1, att2, att3, att4, att5, att6])
        self.fixed_traj_pub.publish(traj)
        self.last_time_pid_pos_publish = rospy.Time.now()

    def spin(self):
        rate = rospy.Rate(self.mpc_rate_hz)
        while not rospy.is_shutdown():
            u_mpc = self.run_mpc()
            u_mpc_norm = self.normalize_control_inputs_mpc(u_mpc.copy())
            self.publish_cmd(u_mpc_norm)
            rate.sleep()


if __name__ == '__main__':
    try:
        node = MPCControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass



 