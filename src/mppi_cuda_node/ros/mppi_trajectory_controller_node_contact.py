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
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseStamped, Vector3Stamped
from tf.transformations import euler_from_quaternion
from scipy.signal import butter


# --- MPPI imports ---
# from mppi_cuda_node.controllers.mppi.mppi_numba_gravity import MPPI_Numba, Config, dynamics_update_sim
# from mppi_cuda_node.controllers.mppi.mppi_numba_gravity_contact import MPPI_Numba, Config, dynamics_update_sim
from mppi_cuda_node.controllers.mppi.mppi_numba_gravity_contact_lcp import MPPI_Numba, Config
# from mppi_cuda_node.controllers.mppi.mppi_numba_gravity_contact import MPPI_Numba, Config, dynamics_update_sim
import mppi_cuda_node.cfg.MPPIParamsConfig as MPPIParamsConfig
from dynamic_reconfigure.server import Server

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
        self.mpc_horizon = 0.8

        self.initialize_hexarotor_parameters()

        # ----- MPPI Setup -----
        self.cfg = Config(
            T=1.0,            # Horizon length in seconds
            dt=0.2,         # Time step (seconds)
            num_control_rollouts=1024*4,
            num_controls=6,
            num_states=12,
            num_vis_state_rollouts=1,
            seed=1
        )
        self.mppi_controller = MPPI_Numba(self.cfg)
        self.use_local_state = False
        self.mppi_params = {
            'dt': self.cfg.dt,
            'x0': self.current_state,
            # Default goal (can be updated via an external command if desired)
            'xgoal': np.array([0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'fgoal': np.array([5, 0, 0]),
            'plane': np.array([1, 0, 0, -1.3]),
            'goal_tolerance': 0.001,
            'dist_weight': 2000,
            'lambda_weight': 10,
            'num_opt': 5,
            'u_std': np.array([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]),
            'vrange': np.array([-10.0, 10.0]),
            'wrange': np.array([-0.1, 0.1]),
            'weights': np.array([
                19550, 19550, 24840,
                1, 1, 1,
                25500, 25500, 25500,
                1, 1, 1,
                1, 100, 1, 100, 200,
                100, 100, 100
            ]),
            "inertia_mass": np.array([self.inertia_flat[0], self.inertia_flat[1], self.inertia_flat[2], self.hex_mass])
        }

        self.current_force_meas = np.zeros(3)
        self.integral_error_z = 0.0  # Initialize integral error for z tracking
        self.I_gain_z = 0.05  # Small integral gain (tune this!)

        self.mppi_controller.set_params(self.mppi_params)
        self.J = np.diag(self.mppi_params['inertia_mass'][:3])

        self.mppi_state = None

        # Prepare an initial control sequence
        self.optimal_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        if GRAVITY:
            # Provide a hover guess for the z-thrust
            self.optimal_control_seq[:, 2] = self.hex_mass * 9.81

        # Optional: a low-pass filter (if you wish to filter commands)
        cutoff_freq = 10
        sampling_rate = 1 / 0.02  # Based on a 50 Hz update rate
        b, a = butter_lowpass_online(cutoff_freq, sampling_rate)
        self.lpf = OnlineLPF(b, a, self.cfg.num_states)

        # ----- Subscribers and Publishers -----
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)
        rospy.Subscriber('/ft_data_filtered', WrenchStamped, self.force_sensor_callback)

        # (Optional: subscribe to an external target command and update self.mppi_params['xgoal'] if needed)

        # Publisher for the target that MPPI computes (for MPC)
        self.target_pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=10)
        self.target_pub_debug = rospy.Publisher('/mppi_debug/target_mpc_debug', PoseStamped, queue_size=10)

        self.mppi_rate_hz = 1/self.cfg.dt  # Run MPPI at 1/dt Hz

        rospy.loginfo("MPPI Controller Node Initialization Complete.")
        # Set up dynamic reconfigure server for tuning MPC parameters
        # self.dyn_server = Server(MPPIParamsConfig, self.dynamic_reconfigure_callback)

        # Deadband
        self.MPPI_mode = np.array(['ON', 'ON', 'ON'], dtype='<U3')
        self.r_on = np.array([0.05, 0.05, 0.05])
        self.r_off = np.array([0.02, 0.02, 0.02])

    def initialize_hexarotor_parameters(self):
        # Set your hexarotor parameters (tweak as needed)
        self.hex_mass = 6.15  # kg (example value)
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        self.inertia_matrix = np.diag(self.inertia_flat)

    def dynamic_reconfigure_callback(self, config, level):
        rospy.loginfo("MPPI Dynamic Reconfigure Request:\n"
                    "dt = %.3f\ngoal_tolerance = %.4f\ndist_weight = %.2f\nlambda_weight = %.2f\nnum_opt = %d\nu_std = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\nweights = [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
                    config['dt'],config['goal_tolerance'],config['dist_weight'],config['lambda_weight'],config['num_opt'],config['u_std_fx'], config['u_std_fy'], config['u_std_fz'], config['u_std_mx'], config['u_std_my'], config['u_std_mz'],config['weights_x'], config['weights_y'], config['weights_z'], config['weights_vx'], config['weights_vy'], config['weights_vz'], config['weights_roll'], config['weights_pitch'], config['weights_yaw'], config['weights_wx'], config['weights_wy'], config['weights_wz'], config['weights_cf'], config['weights_cm'], config['weights_sf'], config['weights_sm'], config['weights_term'])
        
        # Update scalar MPPI parameters
        self.cfg.dt = config['dt']
        # self.mppi_params['dt'] = config['dt']
        self.mppi_params['goal_tolerance'] = config['goal_tolerance']
        self.mppi_params['dist_weight'] = config['dist_weight']
        self.mppi_params['lambda_weight'] = config['lambda_weight']
        self.mppi_params['num_opt'] = config['num_opt']
        
        # Reassemble the u_std array from individual elements.
        self.mppi_params['u_std'] = np.array([ config['u_std_fx'], config['u_std_fy'], config['u_std_fz'], config['u_std_mx'], config['u_std_my'], config['u_std_mz']])
        
        # Reassemble the weights array from individual elements.
        self.mppi_params['weights'] = np.array([ config['weights_x'], config['weights_y'], config['weights_z'], config['weights_vx'], config['weights_vy'], config['weights_vz'], config['weights_roll'], config['weights_pitch'], config['weights_yaw'], config['weights_wx'], config['weights_wy'], config['weights_wz'], config['weights_cf'], config['weights_cm'], config['weights_sf'], config['weights_sm'], config['weights_term']])
    
        # Update the MPPI controller with the new parameters.
        self.mppi_controller.set_params(self.mppi_params)
        
        return config
    

    def force_sensor_callback(self, msg):
        """
        Store the measured 3D force from WrenchStamped.
        Adjust if your sensor orientation is different.
        """
        # Update the current force measurement
        self.current_force_meas[0] = msg.wrench.force.x
        self.current_force_meas[1] = msg.wrench.force.y
        self.current_force_meas[2] = msg.wrench.force.z

    
    def hex_dynamics(self, x, u):
        p, v, Psi, omega = np.split(x, 4)
        f_T, m_T = u[:3], u[3:]
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
            self.mppi_controller.shift_and_update(self.current_state, self.optimal_control_seq, num_shifts=1)
        self.optimal_control_seq = self.mppi_controller.solve()

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
            next_state = self.dynamics_update(self.current_state.copy(), mppi_u, self.mppi_params['dt'])
        # (Optional) Gravity compensation could be applied here if desired.
        # Forward-simulate using a simple RK4 integration:


        # next_state = self.dynamics_update(self.current_state.copy(), mppi_u, self.mppi_params['dt'])
        
        
        # Zero-out the angular velocity components for the target
        # next_state[6:] = np.zeros(6)

        # Periodically update the internal state by blending it with the current state:
        
        # alpha = 0.1  # Adjust this parameter as needed
        # self.mppi_state = alpha * self.current_state + (1 - alpha) * self.mppi_state

        next_state_filtered = self.lpf.filter(next_state)
        next_state_filtered[6:9] = np.clip(next_state_filtered[6:9], -0.02, 0.02)
        # next_state_filtered[2] += self.I_gain_z * self.integral_error_z
        self.mppi_state = next_state_filtered
        self.mpc_target = next_state_filtered


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
        
        # Update per-dimension ON/OFF state
        for i in range(3):  # i=0->x,1->y,2->z
            if self.MPPI_mode[i] == 'OFF':
                # Currently OFF => we only switch ON if we exceed r_on
                if abs(curr[i] - xgoal[i]) > self.r_on[i]:
                    self.MPPI_mode[i] = 'ON'
            elif self.MPPI_mode[i] == 'ON':
                # Currently ON => we switch OFF if we go below r_off
                if abs(curr[i] - xgoal[i]) < self.r_off[i]:
                    self.MPPI_mode[i] = 'OFF'

        # Build the final target state dimension by dimension
        #    If OFF => lock dimension to xgoal, otherwise use next_.
        final_target = np.copy(next_)
        for i in range(3):
            if self.MPPI_mode[i] == 'OFF':
                final_target[i] = xgoal[i]
        

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
