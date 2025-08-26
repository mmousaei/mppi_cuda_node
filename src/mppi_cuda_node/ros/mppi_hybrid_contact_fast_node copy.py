#!/usr/bin/env python
"""
This node runs the MPPI controller:
  - It subscribes to odometry (and an optional activate flag).
  - It computes an optimal control sequence using MPPI.
  - It "forward-simulates" the first control to compute a target state.
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
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseStamped, Vector3Stamped
from tf.transformations import euler_from_quaternion
from scipy.signal import butter
import time

# --- MPPI imports ---
# ONLY CHANGE: Use hybrid contact controller instead of LCP
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from controllers.mppi.mppi_numba_hybrid_contact_fast import MPPI_Numba, Config, dynamics_update_euler
# import mppi_cuda_node.cfg.MPPIParamsConfig as MPPIParamsConfig
# from dynamic_reconfigure.server import Server

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
        self.mpc_force_target = np.zeros(3)
        self.activate = False
        self.mpc_horizon = 0.8
        self.contacting = 0

        self.initialize_hexarotor_parameters()

        # ----- MPPI Setup -----
        self.cfg = Config(
            T=0.6,            # Horizon length in seconds
            dt=0.02,         # Time step (seconds)
            num_control_rollouts=1024*2,
            num_controls=9,
            num_states=15,
            num_vis_state_rollouts=1,
            seed=1
        )
        self.mppi_controller = MPPI_Numba(self.cfg)
        self.use_local_state = False
        self.mppi_params = {
            'dt': self.cfg.dt,
            'x0': np.concatenate((self.current_state, np.array([0, 0, 0]))),
            # Default goal (can be updated via an external command if desired)
            'xgoal': np.array([0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'fgoal': np.array([16, 0, 0]),
            'plane': np.array([-1, 0, 0, 2.2]),
            'goal_tolerance': 0.001,
            'dist_weight': 2000,
            'lambda_weight': 50,
            'num_opt': 5,
            'u_std': np.array([2, 2, 2, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]),
            'vrange': np.array([-10.0, 10.0]),
            'wrange': np.array([-0.1, 0.1]),
            'weights': np.array([
                19550, 19550, 24840,
                1, 1, 1,
                95500, 95500, 95500,
                1, 1, 1,
                1, 100, 1, 100, 200,
                5000, 5000, 5000
            ]),
            "inertia_mass": np.array([self.inertia_flat[0], self.inertia_flat[1], self.inertia_flat[2], self.hex_mass])
        }

        # ----- ROS Setup -----
        self.mppi_rate_hz = 50
        self.target_pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=1)
        self.target_force_pub = rospy.Publisher('/mppi/force_target', WrenchStamped, queue_size=1)
        self.target_pub_debug = rospy.Publisher('/mppi/target_debug', PoseStamped, queue_size=1)
        
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/activate', Bool, self.activate_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)

        # ----- Initialize MPPI Controller -----
        self.mppi_controller.set_params(self.mppi_params)
        self.optimal_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        self.mppi_state = None
        self.integral_error_z = 0.0

        # ----- Deadband Control Setup -----
        self.deadband_indices = [0, 1, 2, 6, 7, 8]  # x, y, z, roll, pitch, yaw
        self.r_on = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])  # Thresholds to turn ON
        self.r_off = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])  # Thresholds to turn OFF
        self.MPPI_mode = ['OFF'] * len(self.deadband_indices)  # Start with all OFF
        self.deadband_timer = np.zeros(len(self.deadband_indices))
        self.deadband_initial_state = np.full(len(self.deadband_indices), np.nan)
        self.deadband_transition_duration = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])  # Transition time in seconds

        # ----- Low-pass filter setup -----
        cutoff_freq = 10.0  # Hz
        b, a = butter_lowpass_online(cutoff_freq, self.mppi_rate_hz, order=1)
        self.lpf = OnlineLPF(b, a, 12)

        # ----- Contact force measurement -----
        self.current_force_meas = np.zeros(3)

        rospy.loginfo("MPPI Controller Node initialized successfully!")

    def initialize_hexarotor_parameters(self):
        """Initialize hexarotor physical parameters"""
        # Mass and inertia
        self.hex_mass = 2.5  # kg
        self.hex_inertia = np.array([0.1, 0.1, 0.2])  # kg*m^2
        self.inertia_flat = self.hex_inertia.flatten()
        
        # Inertia matrix
        self.J = np.diag(self.hex_inertia)

    def hex_dynamics(self, state, control_inputs):
        """
        Hexarotor dynamics model.
        state: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        control_inputs: [fx, fy, fz, mx, my, mz, dfx, dfy, dfz]
        """
        # Extract state variables
        p = state[:3]  # Position
        v = state[3:6]  # Velocity
        Psi = state[6:9]  # Attitude (roll, pitch, yaw)
        omega = state[9:]  # Angular velocity

        # Extract control inputs
        f_T = control_inputs[:3]  # Force in body frame
        m_T = control_inputs[3:6]  # Torque in body frame

        # Rotation matrix from body to world frame
        phi, theta, psi = Psi
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]
        ])
        gravity_world = np.array([0, 0, -9.81])
        gravity_body = np.dot(R.T, gravity_world)  # Rotate gravity to body frame

        nu = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])

        p_dot = v
        v_dot = (1/self.mppi_params['inertia_mass'][3]) * f_T + gravity_body
        psi_dot = np.dot(nu, omega)
        omega_dot = np.dot(np.linalg.inv(self.J), m_T - np.cross(omega, np.dot(self.J, omega)))

        return np.concatenate([p_dot, v_dot, psi_dot, omega_dot])

    def activate_callback(self, data):
        self.activate = data.data
        self.mppi_state = self.current_state
        self.integral_error_z = 0.0

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

        z_error = self.mppi_controller.params['xgoal'][2] - self.current_state[2]  # z_target - z_current
        self.integral_error_z += z_error * self.cfg.dt  # Discrete integration
        self.integral_error_z = np.clip(self.integral_error_z, -0.1, 0.1)  # Tune the range


    def run_mppi(self):
        """
        Run one iteration of MPPI:
         - Shift the previous optimal control sequence.
         - Solve for a new sequence.
        """
        if self.use_local_state:
            if self.mppi_state is None:
                self.mppi_state = self.current_state
                self.mppi_controller.shift_and_update(self.current_state, self.optimal_control_seq, num_shifts=1)
            else:
                self.mppi_controller.shift_and_update(self.mppi_state, self.optimal_control_seq, num_shifts=1)
        else:
            self.mppi_controller.shift_and_update(np.concatenate((self.current_state, self.current_force_meas)), self.optimal_control_seq, num_shifts=1)
        t0 = time.perf_counter()
        self.optimal_control_seq = self.mppi_controller.solve()
        print('solve() took', (time.perf_counter() - t0)*1000, 'ms')

    def forward_simulate_for_mpc_target(self):
        """
        Forward simulate using the first MPPI control (for one time step)
        to obtain a target state that MPC can track.
        """
        mppi_u = self.optimal_control_seq[0, :].copy()
        

        if self.use_local_state:
            forward_steps = max(int(self.mpc_horizon/self.cfg.dt), 1)
            for i in range(forward_steps-1):
                mppi_u = self.optimal_control_seq[i, :].copy()
                self.mppi_state = self.dynamics_update(self.mppi_state.copy(), mppi_u, self.cfg.dt)
            next_state = self.mppi_state.copy()    

        else:

            forward_steps = max(int(self.mpc_horizon/self.cfg.dt/2), 1)
            temp_state = np.concatenate((self.current_state, self.current_force_meas))
            temp_force = np.zeros(3)
            contact = 0
            for i in range(forward_steps-1):
                mppi_u = self.optimal_control_seq[i, :].copy()
                # temp_state, temp_force = self.dynamics_update(temp_state, mppi_u, self.cfg.dt)
                temp_state, temp_force, contact = dynamics_update_euler(temp_state, mppi_u, temp_force, self.cfg.dt, self.mppi_params)
            next_state = temp_state[:12].copy()
            # self.current_force_meas = temp_state[12:]

            # next_state = self.dynamics_update(self.current_state.copy(), mppi_u, self.mppi_params['dt'])
        # (Optional) Gravity compensation could be applied here if desired.
        # Forward-simulate using a simple RK4 integration:


        # next_state = self.dynamics_update(self.current_state.copy(), mppi_u, self.mppi_params['dt'])
        
        
        # Zero-out the angular velocity components for the target
        # next_state[6:] = np.zeros(6)

        # Periodically update the internal state by blending it with the current state:
        
        # alpha = 0.1  # Adjust this parameter as needed
        # self.mppi_state = alpha * self.current_state + (1 - alpha) * self.mppi_state

        next_state_filtered = self.lpf.filter(next_state)
        # next_state_filtered[6:9] = np.clip(next_state_filtered[6:9], -0.02, 0.02)
        # next_state_filtered[2] += self.I_gain_z * self.integral_error_z
        self.mppi_state = next_state_filtered
        self.mpc_target = next_state_filtered
        force_target = np.array([-temp_state[12]*contact, -temp_state[13]*contact, -temp_state[14]*contact])
        self.mpc_force_target = force_target


    def dynamics_update(self, state, control_inputs, dt):
        """
        A simple RK4 integration for the hexarotor dynamics.
        """
        # k1 = dynamics_update_sim(state, control_inputs, dt)
        # k2 = dynamics_update_sim(state + k1 / 2, control_inputs, dt) 
        # k3 = dynamics_update_sim(state + k2 / 2, control_inputs, dt)
        # k4 = dynamics_update_sim(state + k3, control_inputs, dt)
        
        k1 = self.hex_dynamics(state, control_inputs) * dt
        k2 = self.hex_dynamics(state + k1 / 2, control_inputs) * dt 
        k3 = self.hex_dynamics(state + k2 / 2, control_inputs) * dt
        k4 = self.hex_dynamics(state + k3, control_inputs) * dt
        next_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        return next_state
    

    
    def publish_mpc_target(self):
        """
        Publish the computed MPC target as a PoseStamped message.
        (For simplicity, only position is set; orientation is left as a unit quaternion.)
        """
        xgoal = self.mppi_controller.params['xgoal']
        curr  = self.current_state     
        next_ = self.mpc_target        
        
        # Update per-dimension ON/OFF state for the 6 deadband dimensions
        for j, idx in enumerate(self.deadband_indices):
            if self.MPPI_mode[j] == 'OFF':
                # If currently OFF, check if we should re-enable MPPI control
                if abs(curr[idx] - xgoal[idx]) > self.r_on[j]:
                    self.MPPI_mode[j] = 'ON'
            elif self.MPPI_mode[j] == 'ON':
                # If currently ON, switch OFF if error is small
                if abs(curr[idx] - xgoal[idx]) < self.r_off[j]:
                    self.MPPI_mode[j] = 'OFF'

        # Initialize final_target with the current computed target
        final_target = np.copy(next_)
        
        # For each deadband dimension (positions and attitudes)
        for j, idx in enumerate(self.deadband_indices):
            if self.MPPI_mode[j] == 'ON':
                # When MPPI is active, use the computed target.
                final_target[idx] = next_[idx]
                # Reset the deadband timer and initial state.
                self.deadband_timer[j] = 0.0
                self.deadband_initial_state[j] = next_[idx]
            else:
                # When MPPI is off, perform a quadratic ease-out transition toward the final goal.
                if np.isnan(self.deadband_initial_state[j]):
                    self.deadband_initial_state[j] = next_[idx]
                self.deadband_timer[j] += self.cfg.dt
                t = min(self.deadband_timer[j] / self.deadband_transition_duration[j], 1.0)
                # Quadratic ease-out: starts fast and slows down toward the target.
                weight = 1 - (1 - t)**2
                final_target[idx] = (1 - weight) * self.deadband_initial_state[j] + weight * xgoal[idx]
        

        target_msg = PoseStamped()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.pose.position.x = final_target[0]
        target_msg.pose.position.y = final_target[1]
        target_msg.pose.position.z = final_target[2]
        target_msg.pose.orientation.x = final_target[6]
        target_msg.pose.orientation.y = final_target[7]
        target_msg.pose.orientation.z = final_target[8]
        target_msg.pose.orientation.w = 1.0
        self.target_pub.publish(target_msg)
        
        # contacting = np.linalg.norm(self.current_force_meas) > 0.1  # e.g. 1 N
        # if np.linalg.norm(self.current_force_meas) > 0.1:
        #     self.contacting = 1

        if curr[0] > 1.6:
            self.contacting = 1
        else:
            self.contacting = 0 
        if self.contacting:
            # publish the MPPI‐computed force target
            fx, fy, fz = self.mpc_force_target
        else:
            # not touching yet → no force holding
            fx = fy = fz = 0.0

        target_force_msg = WrenchStamped()
        target_force_msg.header.stamp = rospy.Time.now()
        target_force_msg.wrench.force.x = fx
        target_force_msg.wrench.force.y = fy
        target_force_msg.wrench.force.z = fz
        self.target_force_pub.publish(target_force_msg)



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
    def __init__(self, b, a, num_states):
        self.b = b
        self.a = a
        self.prev_input = np.zeros(num_states)
        self.prev_output = np.zeros(num_states)

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
