#!/usr/bin/env python
"""
This node runs the MPPI controller:
  - It subscribes to odometry (and an optional activate flag).
  - It computes an optimal control sequence using MPPI.
  - It “forward-simulates” the first control to compute a target state.
  - It publishes that target as a PoseStamped message on '/mppi/target',
    which the MPC node will subscribe to.
"""

import os
import sys
import numpy as np
import math
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from scipy.signal import butter

# --- MPPI imports ---
from mppi_cuda_node.controllers.mppi.mppi_numba_gravity import MPPI_Numba, Config
# (Assumes that your MPPI_Numba and Config are defined in mppi_numba_gravity.py)

# --- LQR (for dynamics forward-simulation) ---
from mppi_cuda_node.controllers.lqr.lqr_controller import LqrController
from scipy.signal import butter
from scipy.spatial.transform import Rotation

# Global flag (if you want to enable gravity in MPPI)
GRAVITY = True

def butter_lowpass_online(cutoff, fs, order=1):
    """
    Design a low-pass Butterworth filter and return coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

class MPPIControllerNode(object):
    def __init__(self):
        rospy.init_node('mppi_controller', anonymous=True)
        rospy.loginfo("Initializing MPPI Controller Node ...")

        # ----- Initialize state and parameters -----
        self.current_state = np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.mpc_target = np.zeros(12)     # Target state for MPC (to be computed)
        self.activate = False

        self.initialize_hexarotor_parameters()

        # ----- MPPI Setup -----
        self.cfg = Config(
            T=1,            # Horizon length in seconds
            dt=0.2,         # Time step (seconds)
            num_control_rollouts=1024*4,
            num_controls=6,
            num_states=12,
            num_vis_state_rollouts=1,
            seed=1
        )
        self.mppi_controller = MPPI_Numba(self.cfg)

        self.mppi_params = {
            'dt': self.cfg.dt,
            'x0': self.current_state,
            # Default goal (can be updated via an external command if desired)
            'xgoal': np.array([0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'goal_tolerance': 0.001,
            'dist_weight': 2000,
            'lambda_weight': 10,
            'num_opt': 6,
            'u_std': np.array([0.5, 0.5, 0.5, 0.005, 0.005, 0.005]),
            'vrange': np.array([-10.0, 10.0]),
            'wrange': np.array([-0.1, 0.1]),
            'weights': np.array([
                5500, 5500, 3400,
                1, 1, 10,
                800, 800, 800,
                100, 100, 100,
                1, 100, 1, 100, 3000
            ]),
            "inertia_mass": np.array([0.115125971, 0.116524229, 0.230387752, 7.00])
        }
        self.mppi_controller.set_params(self.mppi_params)

        # Prepare an initial control sequence
        self.optimal_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        if GRAVITY:
            # Provide a hover guess for the z-thrust
            self.optimal_control_seq[:, 2] = self.hex_mass * 9.81

        # Optional: a low-pass filter (if you wish to filter commands)
        cutoff_freq = 10
        sampling_rate = 1 / 0.02  # Based on a 50 Hz update rate
        b, a = butter_lowpass_online(cutoff_freq, sampling_rate)
        self.lpf = OnlineLPF(b, a, self.cfg.num_controls)

        # LQR controller for forward-simulation of dynamics
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix

        # ----- Subscribers and Publishers -----
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)
        # (Optional: subscribe to an external target command and update self.mppi_params['xgoal'] if needed)

        # Publisher for the target that MPPI computes (for MPC)
        self.target_pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=10)
        self.target_pub_debug = rospy.Publisher('/mppi_debug/target_mpc_debug', PoseStamped, queue_size=10)

        self.mppi_rate_hz = 5.0  # Run MPPI at 5 Hz

        rospy.loginfo("MPPI Controller Node Initialization Complete.")

    def initialize_hexarotor_parameters(self):
        # Set your hexarotor parameters (tweak as needed)
        self.hex_mass = 7  # kg (example value)
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        self.inertia_matrix = np.diag(self.inertia_flat)

    def activate_callback(self, data):
        self.activate = data.data

    def target_callback(self, data):
        rospy.loginfo("MPPI Target Recieved")
        self.mppi_controller.params['xgoal'] = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
            0, 0, 0,
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            0, 0, 0
        ])

    def odometry_callback(self, data):
        # Update the current state based on odometry
        pose = data.pose.pose
        twist = data.twist.twist

        # Positions and linear velocities
        self.current_state[:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.current_state[3:6] = [twist.linear.x, twist.linear.y, twist.linear.z]

        # Orientation (Euler angles) from quaternion
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        euler = euler_from_quaternion(quaternion)
        self.current_state[6:9] = euler

        # Angular velocities
        self.current_state[9:] = [twist.angular.x, twist.angular.y, twist.angular.z]

    def run_mppi(self):
        """
        Run one iteration of MPPI:
         - Shift the previous optimal control sequence.
         - Solve for a new sequence.
        """
        self.mppi_controller.shift_and_update(self.current_state, self.optimal_control_seq, num_shifts=1)
        self.optimal_control_seq = self.mppi_controller.solve()

    def forward_simulate_for_mpc_target(self):
        """
        Forward simulate using the first MPPI control (for one time step)
        to obtain a target state that MPC can track.
        """
        mppi_u = self.optimal_control_seq[0, :].copy()

        # (Optional) Gravity compensation could be applied here if desired.
        # Forward-simulate using a simple RK4 integration:
        next_state = self.dynamics_update(self.current_state.copy(), mppi_u, self.mppi_params['dt'])
        # Zero-out the angular velocity components for the target
        next_state[6:] = np.zeros(6)
        self.mpc_target = next_state

    def dynamics_update(self, state, control_inputs, dt):
        """
        A simple RK4 integration for the hexarotor dynamics.
        Uses the LQR controller’s dynamics function as an example.
        """
        k1 = self.lqr_controller.hex_dynamics(state, control_inputs) * dt
        k2 = self.lqr_controller.hex_dynamics(state + k1 / 2, control_inputs) * dt
        k3 = self.lqr_controller.hex_dynamics(state + k2 / 2, control_inputs) * dt
        k4 = self.lqr_controller.hex_dynamics(state + k3, control_inputs) * dt
        next_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        return next_state

    def publish_mpc_target(self):
        """
        Publish the computed MPC target as a PoseStamped message.
        (For simplicity, only position is set; orientation is left as a unit quaternion.)
        """
        target_msg = PoseStamped()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.pose.position.x = self.mpc_target[0]
        target_msg.pose.position.y = self.mpc_target[1]
        target_msg.pose.position.z = self.mpc_target[2]
        # Orientation: for now, set to a default value (no rotation)
        target_msg.pose.orientation.x = 0.0
        target_msg.pose.orientation.y = 0.0
        target_msg.pose.orientation.z = 0.0
        target_msg.pose.orientation.w = 1.0
        self.target_pub.publish(target_msg)
        self.target_pub_debug.publish(target_msg)

    def spin(self):
        rate = rospy.Rate(self.mppi_rate_hz)
        while not rospy.is_shutdown():
            
            self.run_mppi()
            self.forward_simulate_for_mpc_target()
            self.publish_mpc_target()
            rate.sleep()


# --- Optional: a simple online low-pass filter class (if needed) ---
class OnlineLPF(object):
    def __init__(self, b, a, num_controls):
        self.b = b
        self.a = a
        self.prev_input = np.zeros(num_controls)
        self.prev_output = np.zeros(num_controls)

    def filter(self, u_curr):
        filtered_u = (self.b[0] * u_curr +
                      self.b[1] * self.prev_input -
                      self.a[1] * self.prev_output)
        self.prev_input = u_curr
        self.prev_output = filtered_u
        return filtered_u


if __name__ == '__main__':
    try:
        node = MPPIControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
