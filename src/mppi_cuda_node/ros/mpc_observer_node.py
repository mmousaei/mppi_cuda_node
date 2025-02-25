#!/usr/bin/env python
"""
This node runs the MPC controller:
  - It subscribes to odometry and a target (from the MPPI/target node).
  - It computes a control command using one-step MPC (or tube MPC).
  - It uses a 6-DOF disturbance observer to estimate model mismatch and provide correction.
  - It publishes control commands and, if activated, sends them via MAVLink.
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
import casadi as cs
import torch

# --- MPC imports ---
from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC
from mppi_cuda_node.controllers.mpc.acados.acados_mpc_tube import TubeMPC
from mppi_cuda_node.controllers.lqr.lqr_controller import LqrController
from mppi_cuda_node.misc.mavlink.mavlink_transmitter import MavlinkTransmitter
from mppi_cuda_node.misc.sim_noise_model.train_noise_model import NoiseNet
from dynamic_reconfigure.server import Server
import mppi_cuda_node.cfg.MPCParamsConfig as MPCParamsConfig


###############################################################################
# 6-DOF Disturbance Observer
###############################################################################
class DisturbanceObserver6D:
    def __init__(self, cutoff_freq, lin_gain, ang_gain, dt, acc_min, acc_max):
        """
        Initialize the 6-DOF disturbance observer.
        
        Args:
            cutoff_freq (float): Cutoff frequency (Hz) for the low-pass filter.
            lin_gain (float): Gain to convert linear acceleration disturbance into force correction.
            ang_gain (float): Gain to convert angular acceleration disturbance into torque correction.
            dt (float): Time step (s).
            acc_min (float or np.array): Minimum allowed disturbance (per channel).
            acc_max (float or np.array): Maximum allowed disturbance (per channel).
        """
        self.cutoff_freq = cutoff_freq
        self.lin_gain = lin_gain
        self.ang_gain = ang_gain
        self.dt = dt
        
        # If scalar limits are provided, create 6-element vectors.
        if np.isscalar(acc_min):
            self.acc_min = np.full(6, acc_min)
        else:
            self.acc_min = np.array(acc_min)
        if np.isscalar(acc_max):
            self.acc_max = np.full(6, acc_max)
        else:
            self.acc_max = np.array(acc_max)
        
        # Initialize previous 6D velocity ([vx, vy, vz, ωx, ωy, ωz])
        self.prev_vel = np.zeros(6)
        
        # Initialize the filtered disturbance (6D)
        self.dist_acc_filt = np.zeros(6)
        
        # Precompute low-pass filter coefficient (first-order filter)
        self.alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_freq * self.dt)

    def update(self, current_vel, desired_acc):
        """
        Update the observer with the current 6D velocity and desired 6D acceleration.
        
        Args:
            current_vel (np.array): 6D measured velocity [vx, vy, vz, ωx, ωy, ωz].
            desired_acc (np.array): 6D desired acceleration [ax, ay, az, αx, αy, αz].
            
        Returns:
            correction (np.array): 6D control correction [Fx, Fy, Fz, τx, τy, τz].
        """
        # Compute actual acceleration as finite difference.
        actual_acc = (current_vel - self.prev_vel) / self.dt
        
        # Disturbance is the difference between measured (actual) and desired acceleration.
        dist_acc = actual_acc - desired_acc
        
        # Apply a first-order low-pass filter.
        self.dist_acc_filt += self.alpha * (dist_acc - self.dist_acc_filt)
        self.dist_acc_filt = np.clip(self.dist_acc_filt, self.acc_min, self.acc_max)
        
        # Convert the filtered disturbance into control correction.
        correction = np.zeros(6)
        correction[0:3] = -self.dist_acc_filt[0:3] * self.lin_gain
        correction[3:6] = -self.dist_acc_filt[3:6] * self.ang_gain
        
        # Update the stored velocity.
        self.prev_vel = current_vel.copy()
        
        return correction


###############################################################################
# MPC Controller Node using the 6-DOF Disturbance Observer
###############################################################################
class MPCControllerNode(object):
    def __init__(self):
        rospy.init_node('mpc_controller', anonymous=True)
        rospy.loginfo("Initializing MPC Controller Node ...")

        # State vectors: 12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.current_state = np.zeros(12)
        self.gt = np.zeros(12)
        self.prev_state = np.zeros(12)
        self.u_mpc = np.zeros(6)
        self.prev_u_mpc = np.zeros(6)
        self.u_total = np.zeros(6)
        self.prev_u_total = np.zeros(6)
        self.mpc_target = np.zeros(12)  # Updated from target messages
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
            'max_torque': 1,
            'control_weight': 0.4,
            'tracking_weight_pos': 50,
            'tracking_weight_vel': 3,
            'tracking_weight_att': 30,
            'tracking_weight_ang_vel': 5,
            'terminal_weight': 1,
            'smoothness_weight': 0.05,
            'dt': 0.01,
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
        
        # ----- 6-DOF Disturbance Observer Initialization -----
        dt = self.mpc_params['dt']
        # For linear correction, we use the vehicle mass.
        lin_gain = 0.1  
        # For angular correction, choose a gain (tune this as needed).
        ang_gain = 0.1  
        # Saturation limits for acceleration disturbance (for all 6 channels)
        acc_min = -1.0  
        acc_max = 1.0  
        cutoff_freq = 1.0  # Hz (tune as needed)
        self.dist_obs6d = DisturbanceObserver6D(cutoff_freq, lin_gain, ang_gain, dt, acc_min, acc_max)

        # LQR controller for auxiliary purposes.
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix

        # ----- Publishers -----
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug', Vector3Stamped, queue_size=10)
        self.fixed_traj_pub = rospy.Publisher("/fixed_trajectory", FixedTrajectory, queue_size=10)

        # ----- MAVLink transmitter -----
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()

        # ----- Subscribers -----
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mpc/target', PoseStamped, self.mpc_target_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)

        self.mpc_rate_hz = 100.0  # Loop frequency in Hz
        self.last_time_pid_pos_publish = rospy.Time.now()

        self.sim = rospy.get_param("/use_sim_time", False)
        self.sim = False
        if self.sim:
            noise_data = np.load("/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src/mppi_cuda_node/src/mppi_cuda_node/misc/sim_noise_model/gaussian_noise.npz")
            self.noise_mean = noise_data["mean"]
            self.noise_cov = noise_data["cov"]

        rospy.loginfo("MPC Controller Node Initialization Complete.")

    def initialize_hexarotor_parameters(self):
        # Set hexarotor parameters (tweak as needed)
        self.hex_mass = 7  # kg (example value)
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        self.inertia_matrix = np.diag(self.inertia_flat)

    def dynamic_reconfigure_callback(self, config, level):
        rospy.loginfo("Reconfigure Request:\nhorizon = %d\ndt = %.3f\nmax_force = %.2f\nmax_torque = %.2f\ncontrol_weight = %.2f\ntracking_weight_pos = %.2f\ntracking_weight_vel = %.2f\ntracking_weight_att = %.2f\ntracking_weight_ang_vel = %.2f\nsmoothness_weight = %.2f",
                      config['horizon'], config['dt'], config['max_force'], config['max_torque'],
                      config['control_weight'], config['tracking_weight_pos'], config['tracking_weight_vel'],
                      config['tracking_weight_att'], config['tracking_weight_ang_vel'], config['smoothness_weight'])
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

    def activate_callback(self, data):
        self.activate = data.data
        # Optionally reset observer state.
        self.dist_obs6d.prev_vel = np.concatenate((self.current_state[3:6], self.current_state[9:12]))

    def odometry_callback(self, data):
        self.prev_state = self.gt.copy()
        self.odom = data
        pose = data.pose.pose
        twist = data.twist.twist

        # Update state: positions and velocities.
        self.current_state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.current_state[3:6] = [twist.linear.x, twist.linear.y, twist.linear.z]

        # Convert quaternion to Euler angles.
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        euler = euler_from_quaternion(quaternion)
        self.current_state[6:9] = euler
        self.current_state[9:] = [twist.angular.x, twist.angular.y, twist.angular.z]

        if self.sim:
            noise_sample = np.random.multivariate_normal(self.noise_mean, self.noise_cov) / 2
            noisy_state = self.current_state + noise_sample.copy()
            self.gt = self.current_state 
        else:
            self.gt = self.current_state

        # Publish attitude debug info.
        att_msg = Vector3Stamped()
        att_msg.header.stamp = data.header.stamp
        att_msg.vector.x = euler[0]
        att_msg.vector.y = euler[1]
        att_msg.vector.z = euler[2]
        self.att_debug_pub.publish(att_msg)

    def mpc_target_callback(self, data):
        self.mpc_target[0] = data.pose.position.x
        self.mpc_target[1] = data.pose.position.y
        self.mpc_target[2] = data.pose.position.z
        self.mpc_target[6] = data.pose.orientation.x
        self.mpc_target[7] = data.pose.orientation.y
        self.mpc_target[8] = data.pose.orientation.z

    def normalize_control_inputs_mpc(self, ctrl):
        """
        Normalize/scaling for MPC outputs (body rates and thrust).
        Adjust these gains to suit your vehicle.
        """
        hover_thrust = 0.6
        ctrl[0] = ctrl[0] * 0.515336334
        ctrl[1] = ctrl[1] * 0.515336334
        ctrl[2] = ctrl[2] * hover_thrust / (self.hex_mass * 9.81)
        ctrl[3:6] = ctrl[3:6] * 0.5
        return ctrl

    def run_mpc(self):
        """
        Solve the one-step MPC using:
         - current_state,
         - mpc_target (set by the target callback),
         - and a zero initial guess.
        """
        mpc_state = self.current_state.copy()
        # Add some noise for robustness.
        # mpc_state[0] += np.random.normal(0.05, 0.01)
        # mpc_state[1] += np.random.normal(-0.05, 0.01)
        u_mpc = self.mpc.compute_control(
            mpc_state,
            self.mpc_target.copy(),
            np.zeros(6),
            self.mpc_params['dt']
        )
        return u_mpc

    def run_mpc_tube(self):
        """
        Compute the control input using tube MPC.
        """
        x_real = self.current_state.copy()
        x_nom = self.mpc_target.copy()
        u_hover = np.array([0.0, 0.0, self.hex_mass * 9.81, 0.0, 0.0, 0.0])
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

        # Prepare attitude command parameters for the transmitter.
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
        Publish a fixed trajectory for debugging.
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
            self.prev_u_mpc = self.u_mpc.copy()
            self.prev_u_total = self.u_total.copy()
            self.u_mpc = self.run_mpc()
            
            # Construct the 6D measured velocity vector: [vx, vy, vz, ωx, ωy, ωz]
            current_vel_6d = np.concatenate((self.current_state[3:6], self.current_state[9:12]))
            
            # Compute the desired acceleration for each channel.
            # For linear channels, desired acceleration = force/mass.
            desired_lin_acc = self.u_mpc[0:3] / self.hex_mass
            # For angular channels, desired acceleration = torque / inertia (elementwise division).
            desired_ang_acc = self.u_mpc[3:6] / self.inertia_flat
            desired_acc_6d = np.concatenate((desired_lin_acc, desired_ang_acc))
            
            # Get the 6-DOF correction from the disturbance observer.
            correction = self.dist_obs6d.update(current_vel_6d, desired_acc_6d)
            
            # Add the correction to the MPC command.
            self.u_total = self.u_mpc + correction
            u_total_norm = self.normalize_control_inputs_mpc(self.u_total.copy())
            self.publish_cmd(u_total_norm)
            rate.sleep()


if __name__ == '__main__':
    try:
        node = MPCControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
