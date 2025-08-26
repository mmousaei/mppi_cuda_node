#!/usr/bin/env python
"""
MPPI Controller Node with Hybrid Contact Dynamics

This node runs the MPPI controller with improved contact modeling:
  - Hybrid impulse-force contact dynamics for numerical stability
  - Smooth contact activation to avoid binary transitions
  - Adaptive stiffness and damping for better physical consistency
  - Improved integration with MPC for contact force tracking

Features:
  - Smooth contact detection with configurable transition zones
  - Adaptive contact parameters based on penetration depth and velocity
  - Friction modeling for realistic contact behavior
  - Force rate control for smooth force transitions
  - Enhanced stability during MPPI sampling
"""

import os
import sys
import numpy as np
import math
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, WrenchStamped, Vector3Stamped
from tf.transformations import euler_from_quaternion
from scipy.signal import butter
import time

# --- MPPI imports ---
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from controllers.mppi.mppi_numba_hybrid_contact import MPPI_Numba, Config, dynamics_update_euler
import mppi_cuda_node.cfg.MPPIParamsConfig as MPPIParamsConfig
from dynamic_reconfigure.server import Server

from scipy.signal import butter
from scipy.spatial.transform import Rotation

# Global flag for gravity compensation
GRAVITY = True

def butter_lowpass_online(cutoff, fs, order=1):
    """Design a low-pass Butterworth filter and return coefficients."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

class OnlineLPF(object):
    """Online low-pass filter for smoothing control outputs."""
    
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

class MPPIHybridContactNode(object):
    """MPPI Controller Node with Hybrid Contact Dynamics"""
    
    def __init__(self):
        rospy.init_node('mppi_hybrid_contact_controller', anonymous=True)
        rospy.loginfo("Initializing MPPI Hybrid Contact Controller Node ...")

        # ----- Initialize state and parameters -----
        self.current_state = np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.mpc_target = np.zeros(12)     # Target state for MPC
        self.mpc_force_target = np.zeros(3)  # Target contact forces for MPC
        self.activate = False
        self.mpc_horizon = 0.8
        self.contacting = 0
        self.contact_confidence = 0.0  # Smooth contact state (0-1)

        self.initialize_hexarotor_parameters()
        self.initialize_contact_parameters()

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

        self.current_force_meas = np.zeros(3)
        self.integral_error_z = 0.0
        self.I_gain_z = 0.05

        self.mppi_controller.set_params(self.mppi_params)
        self.J = np.diag(self.mppi_params['inertia_mass'][:3])

        self.mppi_state = None

        # Prepare initial control sequence
        self.optimal_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        if GRAVITY:
            self.optimal_control_seq[:, 2] = self.hex_mass * 9.81

        # Low-pass filter for command smoothing
        cutoff_freq = 10
        sampling_rate = 1 / 0.02
        b, a = butter_lowpass_online(cutoff_freq, sampling_rate)
        self.lpf = OnlineLPF(b, a, self.cfg.num_states-3)

        # ----- Subscribers and Publishers -----
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)
        rospy.Subscriber('/ft_data_filtered', WrenchStamped, self.force_sensor_callback)

        # Publishers
        self.target_pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=10)
        self.target_force_pub = rospy.Publisher('/mpc/wrenchtarget', WrenchStamped, queue_size=10)
        self.target_pub_debug = rospy.Publisher('/mppi_debug/target_mpc_debug', PoseStamped, queue_size=10)
        self.contact_state_pub = rospy.Publisher('/mppi_debug/contact_state', Vector3Stamped, queue_size=10)

        self.mppi_rate_hz = 1/self.cfg.dt

        # ----- Contact State Management -----
        self.contact_transition_timer = 0.0
        self.contact_transition_duration = 0.5  # seconds
        self.last_contact_state = False
        
        # ----- Deadband Control -----
        self.deadband_indices = [0, 1, 2, 6, 7, 8]
        self.MPPI_mode = np.array(['ON'] * 6, dtype='<U3')
        self.r_on = np.array([0.1, 0.1, 0.1, 0.08, 0.08, 0.08])
        self.r_off = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.deadband_timer = np.zeros(6)
        self.deadband_initial_state = np.full(6, np.nan)
        self.deadband_transition_duration = np.array([1]*3 + [3]*3)

        rospy.loginfo("MPPI Hybrid Contact Controller Node Initialization Complete.")

    def initialize_hexarotor_parameters(self):
        """Initialize hexarotor physical parameters."""
        self.hex_mass = 6.15  # kg
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        self.inertia_matrix = np.diag(self.inertia_flat)

    def initialize_contact_parameters(self):
        """Initialize contact dynamics parameters."""
        # Contact detection parameters
        self.contact_threshold = 0.02  # meters - distance threshold for contact detection
        self.contact_transition_distance = 0.02  # meters - smooth transition zone
        
        # Contact force parameters
        self.max_contact_force = 50.0  # N - maximum allowed contact force
        self.force_filter_cutoff = 20.0  # Hz - low-pass filter for contact forces
        
        # Contact state hysteresis
        self.contact_hysteresis_on = 0.05   # meters - contact activation threshold
        self.contact_hysteresis_off = 0.02  # meters - contact deactivation threshold

    def dynamic_reconfigure_callback(self, config, level):
        """Dynamic reconfigure callback for MPPI parameters."""
        rospy.loginfo("MPPI Dynamic Reconfigure Request:\n"
                    "dt = %.3f\ngoal_tolerance = %.4f\ndist_weight = %.2f\nlambda_weight = %.2f\nnum_opt = %d\nu_std = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\nweights = [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
                    config['dt'],config['goal_tolerance'],config['dist_weight'],config['lambda_weight'],config['num_opt'],config['u_std_fx'], config['u_std_fy'], config['u_std_fz'], config['u_std_mx'], config['u_std_my'], config['u_std_mz'],config['weights_x'], config['weights_y'], config['weights_z'], config['weights_vx'], config['weights_vy'], config['weights_vz'], config['weights_roll'], config['weights_pitch'], config['weights_yaw'], config['weights_wx'], config['weights_wy'], config['weights_wz'], config['weights_cf'], config['weights_cm'], config['weights_sf'], config['weights_sm'], config['weights_term'])
        
        # Update scalar MPPI parameters
        self.cfg.dt = config['dt']
        self.mppi_params['goal_tolerance'] = config['goal_tolerance']
        self.mppi_params['dist_weight'] = config['dist_weight']
        self.mppi_params['lambda_weight'] = config['lambda_weight']
        self.mppi_params['num_opt'] = config['num_opt']
        
        # Reassemble arrays from individual elements
        self.mppi_params['u_std'] = np.array([config['u_std_fx'], config['u_std_fy'], config['u_std_fz'], config['u_std_mx'], config['u_std_my'], config['u_std_mz']])
        # Reassemble weights array from individual elements
        weights_array = [
            config['weights_x'], config['weights_y'], config['weights_z'],
            config['weights_vx'], config['weights_vy'], config['weights_vz'],
            config['weights_roll'], config['weights_pitch'], config['weights_yaw'],
            config['weights_wx'], config['weights_wy'], config['weights_wz'],
            config['weights_cf'], config['weights_cm'], config['weights_sf'],
            config['weights_sm'], config['weights_term']
        ]
        self.mppi_params['weights'] = np.array(weights_array)
        
        # Update the MPPI controller
        self.mppi_controller.set_params(self.mppi_params)
        
        return config

    def force_sensor_callback(self, msg):
        """Store the measured 3D force from WrenchStamped."""
        self.current_force_meas[0] = -msg.wrench.force.x
        self.current_force_meas[1] = -msg.wrench.force.y
        self.current_force_meas[2] = -msg.wrench.force.z

    def hex_dynamics(self, x, u):
        """Simple hexarotor dynamics for fallback simulation."""
        p, v, Psi, omega = np.split(x, 4)
        f_T, m_T = u[:3], u[3:]
        phi, theta, psi = Psi
        
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]
        ])
        
        gravity_world = np.array([0, 0, -9.81])
        gravity_body = np.dot(R.T, gravity_world)

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
        """Callback for MPPI activation."""
        self.activate = data.data
        self.mppi_state = self.current_state
        self.integral_error_z = 0.0

    def target_callback(self, data):
        """Callback for external target commands."""
        rospy.loginfo("MPPI Target Received")
        # Update the goal state in MPPI parameters
        if hasattr(self.mppi_controller, 'params') and self.mppi_controller.params is not None:
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
        """Callback for odometry data."""
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

        # Integral error for z tracking
        if hasattr(self.mppi_controller, 'params') and self.mppi_controller.params is not None:
            z_error = self.mppi_controller.params['xgoal'][2] - self.current_state[2]
            self.integral_error_z += z_error * self.cfg.dt
            self.integral_error_z = np.clip(self.integral_error_z, -0.1, 0.1)

    def update_contact_state(self):
        """Update contact state with hysteresis and smooth transitions."""
        # Calculate distance to contact surface
        plane = self.mppi_params['plane']
        A, B, C, D = plane
        x, y, z = self.current_state[:3]
        
        # Distance to plane (positive = above surface, negative = below surface)
        distance_to_surface = (A*x + B*y + C*z + D) / math.sqrt(A*A + B*B + C*C)
        
        # Contact detection with hysteresis
        if distance_to_surface < -self.contact_hysteresis_on:
            target_contact_state = True
        elif distance_to_surface > -self.contact_hysteresis_off:
            target_contact_state = False
        else:
            # In hysteresis zone - maintain current state
            target_contact_state = self.last_contact_state
        
        # Smooth contact state transition
        if target_contact_state != self.last_contact_state:
            self.contact_transition_timer = 0.0
            self.last_contact_state = target_contact_state
        
        # Update contact confidence with smooth transition
        if self.contact_transition_timer < self.contact_transition_duration:
            self.contact_transition_timer += self.cfg.dt
            transition_progress = self.contact_transition_timer / self.contact_transition_duration
            
            if target_contact_state:
                # Transitioning to contact
                self.contact_confidence = transition_progress
            else:
                # Transitioning away from contact
                self.contact_confidence = 1.0 - transition_progress
        else:
            # Transition complete
            self.contact_confidence = 1.0 if target_contact_state else 0.0
        
        # Update contacting flag for backward compatibility
        self.contacting = 1 if self.contact_confidence > 0.5 else 0
        
        return distance_to_surface

    def run_mppi(self):
        """Run one iteration of MPPI optimization."""
        try:
            # Update MPPI parameters with current state
            print("run_mppi")
            current_state_extended = np.concatenate((self.current_state, self.current_force_meas))
            self.mppi_params['x0'] = current_state_extended
            
            # Update the controller parameters
            print("set_params")
            self.mppi_controller.set_params(self.mppi_params)
            
            # Run MPPI optimization
            print("solve")
            self.optimal_control_seq = self.mppi_controller.solve()
            print("solve done")
            rospy.logdebug("MPPI optimization completed successfully")
            
        except Exception as e:
            rospy.logerr(f"MPPI optimization failed: {e}")
            # Fallback to previous control sequence if available
            if self.optimal_control_seq is not None:
                rospy.logwarn("Using previous control sequence as fallback")

    def forward_simulate_for_mpc_target(self):
        """Forward simulate using MPPI controls to obtain MPC target."""
        try:
            if self.optimal_control_seq is None or len(self.optimal_control_seq) == 0:
                rospy.logwarn("No optimal control sequence available for forward simulation")
                return
            
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
                    temp_state, temp_force, contact = dynamics_update_euler(temp_state, mppi_u, temp_force, self.cfg.dt, self.mppi_params)
                
                next_state = temp_state[:12].copy()

            # Apply low-pass filtering
            next_state_filtered = self.lpf.filter(next_state)
            self.mppi_state = next_state_filtered
            self.mpc_target = next_state_filtered
            
            # Compute force target for MPC
            force_target = np.array([-temp_state[12]*contact, -temp_state[13]*contact, -temp_state[14]*contact])
            self.mpc_force_target = force_target
            
        except Exception as e:
            rospy.logerr(f"Forward simulation failed: {e}")

    def dynamics_update(self, state, control_inputs, dt):
        """Simple RK4 integration for hexarotor dynamics."""
        k1 = self.hex_dynamics(state, control_inputs) * dt
        k2 = self.hex_dynamics(state + k1 / 2, control_inputs) * dt 
        k3 = self.hex_dynamics(state + k2 / 2, control_inputs) * dt
        k4 = self.hex_dynamics(state + k3, control_inputs) * dt
        next_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        return next_state

    def publish_mpc_target(self):
        """Publish the computed MPC target as a PoseStamped message."""
        try:
            print("publish_mpc_target")
            if not hasattr(self.mppi_controller, 'params') or self.mppi_controller.params is None:
                rospy.logwarn("MPPI parameters not available")
                return
                
            xgoal = self.mppi_controller.params['xgoal']
            curr = self.current_state     
            next_ = self.mpc_target        
            
            # Update per-dimension ON/OFF state for deadband control
            for j, idx in enumerate(self.deadband_indices):
                if self.MPPI_mode[j] == 'OFF':
                    if abs(curr[idx] - xgoal[idx]) > self.r_on[j]:
                        self.MPPI_mode[j] = 'ON'
                elif self.MPPI_mode[j] == 'ON':
                    if abs(curr[idx] - xgoal[idx]) < self.r_off[j]:
                        self.MPPI_mode[j] = 'OFF'

            # Initialize final target
            final_target = np.copy(next_)
            
            # Apply deadband control
            for j, idx in enumerate(self.deadband_indices):
                if self.MPPI_mode[j] == 'ON':
                    final_target[idx] = next_[idx]
                    self.deadband_timer[j] = 0.0
                    self.deadband_initial_state[j] = next_[idx]
                else:
                    if np.isnan(self.deadband_initial_state[j]):
                        self.deadband_initial_state[j] = next_[idx]
                    self.deadband_timer[j] += self.cfg.dt
                    t = min(self.deadband_timer[j] / self.deadband_transition_duration[j], 1.0)
                    weight = 1 - (1 - t)**2
                    final_target[idx] = (1 - weight) * self.deadband_initial_state[j] + weight * xgoal[idx]

            # Publish target pose
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

            # Publish force target
            if self.contacting:
                fx, fy, fz = self.mpc_force_target
            else:
                fx = fy = fz = 0.0

            target_force_msg = WrenchStamped()
            target_force_msg.header.stamp = rospy.Time.now()
            target_force_msg.wrench.force.x = fx
            target_force_msg.wrench.force.y = fy
            target_force_msg.wrench.force.z = fz
            self.target_force_pub.publish(target_force_msg)

            # Publish debug information
            self.target_pub_debug.publish(target_msg)
            
            # Publish contact state for debugging
            contact_state_msg = Vector3Stamped()
            contact_state_msg.header.stamp = rospy.Time.now()
            contact_state_msg.vector.x = self.contact_confidence
            contact_state_msg.vector.y = self.contacting
            contact_state_msg.vector.z = self.update_contact_state()  # Distance to surface
            self.contact_state_pub.publish(contact_state_msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish MPC target: {e}")

    def spin(self):
        """Main control loop."""
        rate = rospy.Rate(self.mppi_rate_hz)
        while not rospy.is_shutdown():
            try:
                if self.activate:
                    self.run_mppi()
                    self.forward_simulate_for_mpc_target()
                    self.publish_mpc_target()
                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error in main control loop: {e}")
                rate.sleep()

if __name__ == '__main__':
    try:
        node = MPPIHybridContactNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
