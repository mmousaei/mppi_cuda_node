#!/usr/bin/env python
"""
This node runs the MPC controller with full EKF fusion including angular velocity:
  - It subscribes to high-rate IMU raw data for prediction.
  - It subscribes to lower-rate odometry for position, velocity, attitude, and angular velocity updates.
  - It fuses these measurements using an Extended Kalman Filter.
  - It computes control commands using the fused state.
  - It compensates the MPC command using a model mismatch observer implemented as a second EKF.
  - It publishes control commands and (if activated) sends them via MAVLink.
"""

import os
import sys
import numpy as np
import math
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
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
from scipy.spatial.transform import Rotation
from mppi_cuda_node.misc.sim_noise_model.train_noise_model import NoiseNet

from dynamic_reconfigure.server import Server
import mppi_cuda_node.cfg.MPCParamsConfig as MPCParamsConfig


##############################################
#        Extended Kalman Filter Class        #
##############################################
class EKF(object):
    def __init__(self):
        # State: [p (3), v (3), rpy (3), omega (3)] -> 12 dimensions.
        self.x = np.zeros(12)
        self.P = np.eye(12) * 0.1

        # Process noise covariance (tune as needed)
        self.Q = np.eye(12) * 0.01

        # Measurement noise covariance (tune as needed)
        self.R = np.eye(12) * 0.05

    def euler_to_rot(self, rpy):
        roll, pitch, yaw = rpy
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll),  np.cos(roll)]])
        R_y = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                        [0,              1,             0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0,                      0, 1]])
        return R_z.dot(R_y).dot(R_x)

    def f(self, x, u, dt):
        """
        Process model:
          x = [p, v, rpy, omega]
          u = [acc (3), gyro (3)]
        Prediction uses:
          - p: p + dt*v
          - v: v + dt*(R(rpy)*acc + g)
          - rpy: rpy + dt*gyro   (using IMU gyro for fast attitude update)
          - omega: set to gyro (IMU angular velocity)
        """
        p = x[0:3]
        v = x[3:6]
        rpy = x[6:9]
        acc = u[0:3]
        gyro = u[3:6]
        R_mat = self.euler_to_rot(rpy)
        g = np.array([0, 0, -9.81])
        p_pred = p + dt * v
        v_pred = v + dt * (R_mat.dot(acc) + g)
        rpy_pred = rpy + dt * gyro
        omega_pred = gyro
        return np.concatenate([p_pred, v_pred, rpy_pred, omega_pred])

    def compute_F(self, x, u, dt, epsilon=1e-5):
        n = x.shape[0]
        F = np.zeros((n, n))
        f0 = self.f(x, u, dt)
        for i in range(n):
            x_eps = x.copy()
            x_eps[i] += epsilon
            f_eps = self.f(x_eps, u, dt)
            F[:, i] = (f_eps - f0) / epsilon
        return F

    def predict(self, u, dt):
        F = self.compute_F(self.x, u, dt)
        self.x = self.f(self.x, u, dt)
        self.P = F.dot(self.P).dot(F.T) + self.Q

    def update(self, z):
        H = np.eye(12)  # Direct measurement of state
        y = z - self.x
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(12) - K.dot(H)).dot(self.P)


##############################################
#          MPC Controller Node Class         #
##############################################
class MPCControllerNode(object):
    def __init__(self):
        rospy.init_node('mpc_controller', anonymous=True)
        rospy.loginfo("Initializing MPC Controller Node with EKF (including angular velocity)...")

        # Initialize the primary EKF for sensor fusion.
        self.ekf = EKF()
        self.last_imu_time = None  # To compute dt in prediction

        # Other state variables.
        self.u_mpc = np.zeros(6)
        self.u_total = np.zeros(6)
        self.mpc_target = np.zeros(12)  # To be updated from the MPPI node
        self.mpc_target[2] = 0.8
        self.activate = False
        self.odom = None  # Will be set in the odometry callback

        self.initialize_hexarotor_parameters()

        # ----- MPC Setup -----
        self.mpc_l1_params = {
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
            # For the low-pass filtering of the compensation output,
            # we now define separate time constants:
            'l1_tau_trans': 0.1,  # translational time constant (seconds)
            'l1_tau_rot': 0.05    # rotational time constant (seconds)
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
        self.mpc = MPC(self.mpc_l1_params)
        self.tube_mpc = TubeMPC(self.mpc_tube_params)
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix

        # Create a CasADi function for the nominal dynamics.
        # (This is the same nominal model used in your MPC.)
        self.f_nominal = cs.Function('f_nominal', [self.mpc.model.x, self.mpc.model.u],
                                       [self.mpc.model.f_expl_expr])

        # ----- Set up the model mismatch observer as a second EKF.
        self.mismatch_ekf = EKF()
        # Initialize the mismatch EKF state to match the primary EKF.
        self.mismatch_ekf.x = self.ekf.x.copy()
        # Override its process model to use the nominal dynamics.
        self.mismatch_ekf.f = lambda x, u, dt: np.array(self.f_nominal(x, u)).flatten()

        # Compute B and its pseudoinverse for mapping state error to control channels.
        self.B = np.zeros((12, 6))
        mass = self.hex_mass
        I_xx, I_yy, I_zz = self.inertia_flat
        self.B[3, 0] = 1.0 / mass
        self.B[4, 1] = 1.0 / mass
        self.B[5, 2] = 1.0 / mass
        self.B[9, 3] = 1.0 / I_xx
        self.B[10, 4] = 1.0 / I_yy
        self.B[11, 5] = 1.0 / I_zz
        self.B_pinv = np.linalg.pinv(self.B)

        # Low-pass filter state for the compensation command.
        self.u_comp_filtered = np.zeros(6)

        # ----- Publishers -----
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.control_pub_umpc = rospy.Publisher('/mppi_debug/control_cmd_umpc', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug', Vector3Stamped, queue_size=10)
        self.fixed_traj_pub = rospy.Publisher("/fixed_trajectory", FixedTrajectory, queue_size=10)

        # ----- MAVLink transmitter -----
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()

        # ----- Subscribers -----
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mavros/imu/data_raw', Imu, self.imu_callback)
        rospy.Subscriber('/mpc/target', PoseStamped, self.mpc_target_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)

        # Run the controller loop at a high rate if possible.
        self.mpc_rate_hz = 200.0  # Adjust based on your hardware capabilities.
        self.last_time_pid_pos_publish = rospy.Time.now()
        self.sim = rospy.get_param("/use_sim_time", False)

        rospy.loginfo("MPC Controller Node with EKF Initialization Complete.")

    def initialize_hexarotor_parameters(self):
        self.hex_mass = 7  # kg (example value)
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        self.inertia_matrix = np.diag(self.inertia_flat)

    def imu_callback(self, data):
        """
        IMU callback: use raw IMU data to run the primary EKF prediction step.
        u = [acceleration (3), gyro (3)]
        """
        acc = np.array([data.linear_acceleration.x,
                        data.linear_acceleration.y,
                        data.linear_acceleration.z])
        gyro = np.array([data.angular_velocity.x,
                         data.angular_velocity.y,
                         data.angular_velocity.z])
        u = np.concatenate([acc, gyro])
        now = data.header.stamp.to_sec()
        if self.last_imu_time is None:
            dt = 1.0 / self.mpc_rate_hz
        else:
            dt = now - self.last_imu_time
        self.last_imu_time = now

        # Primary EKF prediction.
        self.ekf.predict(u, dt)

        # Optionally publish fused attitude for debugging.
        fused_rpy = self.ekf.x[6:9]
        att_msg = Vector3Stamped()
        att_msg.header.stamp = data.header.stamp
        att_msg.vector.x = fused_rpy[0]
        att_msg.vector.y = fused_rpy[1]
        att_msg.vector.z = fused_rpy[2]
        self.att_debug_pub.publish(att_msg)

    def odometry_callback(self, data):
        """
        Odometry callback: update the primary EKF with measured state.
        """
        self.odom = data
        p = np.array([data.pose.pose.position.x,
                      data.pose.pose.position.y,
                      data.pose.pose.position.z])
        v = np.array([data.twist.twist.linear.x,
                      data.twist.twist.linear.y,
                      data.twist.twist.linear.z])
        quaternion = [data.pose.pose.orientation.x,
                      data.pose.pose.orientation.y,
                      data.pose.pose.orientation.z,
                      data.pose.pose.orientation.w]
        rpy = np.array(euler_from_quaternion(quaternion))
        omega = np.array([data.twist.twist.angular.x,
                          data.twist.twist.angular.y,
                          data.twist.twist.angular.z])
        z = np.concatenate([p, v, rpy, omega])
        self.ekf.update(z)

    def mpc_target_callback(self, data):
        """
        Callback for the target published by the MPPI node.
        Updates our internal MPC target state.
        """
        self.mpc_target[0] = data.pose.position.x
        self.mpc_target[1] = data.pose.position.y
        self.mpc_target[2] = data.pose.position.z
        # Use target attitude directly.
        self.mpc_target[6] = data.pose.orientation.x
        self.mpc_target[7] = data.pose.orientation.y
        self.mpc_target[8] = data.pose.orientation.z
        
    def activate_callback(self, data):
        self.activate = data.data
        # Reset the mismatch observer EKF and its filtered compensation.
        self.mismatch_ekf.x = self.ekf.x.copy()
        self.u_comp_filtered = np.zeros(6)

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
        Solve the one-step MPC using the fused state from the primary EKF.
        """
        fused_state = self.ekf.x.copy()
        u_mpc = self.mpc.compute_control(
            fused_state.copy(),
            self.mpc_target.copy(),
            np.zeros(6),
            self.mpc_l1_params['dt']
        )
        return u_mpc

    def model_mismatch_observer(self, measured_state, u_mpc, dt):
        """
        Model mismatch observer using a second EKF:
          - Predicts the observer state using the nominal model.
          - Updates the observer with the measured state.
          - The mismatch error is computed as the difference between the measured state and the observer state.
          - This error is mapped to control space via B_pinv and low-pass filtered separately for translation and rotation.
        """
        # Run prediction and update for the mismatch EKF.
        self.mismatch_ekf.predict(u_mpc, dt)
        self.mismatch_ekf.update(measured_state)
        
        # Compute the error between the measured state and the mismatch observer's state.
        error = measured_state - self.mismatch_ekf.x
        
        # Map the error into the 6-D control space.
        delta = self.B_pinv @ error
        
        # Apply separate low-pass filtering to translational (0:3) and rotational (3:6) parts.
        tau_trans = self.mpc_l1_params.get('l1_tau_trans', 0.1)
        tau_rot   = self.mpc_l1_params.get('l1_tau_rot', 0.05)
        alpha_trans = dt / (dt + tau_trans)
        alpha_rot   = dt / (dt + tau_rot)
        
        self.u_comp_filtered[0:3] = (1 - alpha_trans) * self.u_comp_filtered[0:3] + alpha_trans * (-delta[0:3])
        self.u_comp_filtered[3:6] = (1 - alpha_rot)   * self.u_comp_filtered[3:6]   + alpha_rot   * (-delta[3:6])
        
        # Apply saturation limits.
        fmax = self.mpc_l1_params.get("max_force", 10.0)
        mmax = self.mpc_l1_params.get("max_torque", 1.0)
        sigma_max = np.concatenate([np.array([fmax]*3), np.array([mmax]*3)])
        u_comp = np.clip(self.u_comp_filtered, -sigma_max, sigma_max)
        
        return u_comp

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
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        z = self.odom.pose.pose.position.z

        traj = FixedTrajectory()
        traj.type = "Point"
        traj.attributes.extend([
            KeyValue(key="frame_id", value="world"),
            KeyValue(key="height", value=str(z)),
            KeyValue(key="max_acceleration", value=str(0.4)),
            KeyValue(key="velocity", value=str(0.1)),
            KeyValue(key="x", value=str(x)),
            KeyValue(key="y", value=str(y))
        ])
        self.fixed_traj_pub.publish(traj)
        self.last_time_pid_pos_publish = rospy.Time.now()

    def publish_umpc_uadapt_debug(self, u_mpc, u_comp):
        cmd_msg = WrenchStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.wrench.force.x = u_mpc[0]
        cmd_msg.wrench.force.y = u_mpc[1]
        cmd_msg.wrench.force.z = u_mpc[2]
        cmd_msg.wrench.torque.x = u_mpc[3]
        cmd_msg.wrench.torque.y = u_mpc[4]
        cmd_msg.wrench.torque.z = u_mpc[5]
        self.control_pub_umpc.publish(cmd_msg)
        # (Optionally, publish u_comp separately for debugging.)

    def spin(self):
        rate = rospy.Rate(self.mpc_rate_hz)
        while not rospy.is_shutdown():
            dt = 1.0 / self.mpc_rate_hz
            self.u_mpc = self.run_mpc()
            # Compute the model mismatch compensation using the second EKF.
            u_comp = self.model_mismatch_observer(self.ekf.x.copy(), self.u_mpc.copy(), dt)
            self.publish_umpc_uadapt_debug(self.u_mpc.copy(), u_comp.copy())
            # Combine the MPC command with the compensation.
            self.u_total = self.u_mpc + u_comp
            u_total_norm = self.normalize_control_inputs_mpc(self.u_total.copy())
            self.publish_cmd(u_total_norm)
            rate.sleep()


if __name__ == '__main__':
    try:
        node = MPCControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
