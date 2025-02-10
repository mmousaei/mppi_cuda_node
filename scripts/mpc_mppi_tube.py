#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import numpy as np
import rospy

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from mavlink_transmitter import MavlinkTransmitter
from geometry_msgs.msg import Vector3Stamped
from core_trajectory_msgs.msg import FixedTrajectory
from diagnostic_msgs.msg import KeyValue

# --- MPPI and MPC imports ---
from mppi_numba_gravity import MPPI_Numba, Config
from acados_mpc import OneStepMPC
from acados_mpc_tube import TubeMPC

from lqr_controller import LqrController
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
from scipy.signal import butter

GRAVITY = True

def butter_lowpass_online(cutoff, fs, order=1):
    """
    Design a low-pass Butterworth filter and return coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

class OnlineLPF:
    def __init__(self, b, a, num_controls):
        """
        Initialize the LPF with given coefficients and number of controls.
        """
        self.b = b
        self.a = a
        self.prev_input = np.zeros(num_controls)
        self.prev_output = np.zeros(num_controls)

    def filter(self, u_curr):
        """
        Apply the LPF to the current control inputs.
        """
        filtered_u = (
            self.b[0] * u_curr +
            self.b[1] * self.prev_input -
            self.a[1] * self.prev_output
        )

        self.prev_input = u_curr
        self.prev_output = filtered_u
        return filtered_u

class ControlHexarotor:
    def __init__(self):
        rospy.init_node('hexarotor_controller', anonymous=True)
        print("Initializing Hexarotor Controller ...")

        self.integral_error_z = 0.0  
        self.last_time = None
        self.Ki_z = 0.1

        self.initialize_hexarotor_parameters()
        
        # ------------------------------
        # Publishers/Subscribers
        # ------------------------------
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug', Vector3Stamped, queue_size=10)
        self.target_debug_pub = rospy.Publisher('/mppi_debug/target_debug', PoseStamped, queue_size=10)
        self.target_mpc_debug_pub = rospy.Publisher('/mppi_debug/target_mpc_debug', PoseStamped, queue_size=10)
        self.fixed_traj_pub = rospy.Publisher("/fixed_trajectory", FixedTrajectory, queue_size=10)

        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mppi/target', PoseStamped, self.target_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)

        self.current_state = np.zeros(12)  
        self.control_inputs = WrenchStamped()
        self.odom = Odometry()
        self.last_time_pid_pos_publish = rospy.Time.now()

        self.activate = False
        self.cnt = 0

        # ------------------------------
        # MPPI Setup
        # ------------------------------
        constant = 2
        self.cfg = Config(
            T=2,                # Horizon length in seconds
            dt=0.2,        # Time step
            num_control_rollouts=1024*8,
            num_controls=6,
            num_states=12,
            num_vis_state_rollouts=1,
            seed=1
        )

        cutoff_freq = 10
        sampling_rate = 1 / 0.02  # Based on the time step in MPPI
        b, a = butter_lowpass_online(cutoff_freq, sampling_rate)
        self.lpf = OnlineLPF(b, a, self.cfg.num_controls)

        self.optimal_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        if GRAVITY:
            # Provide a hover guess for the z-thrust
            self.optimal_control_seq[:, 2] = self.hex_mass * 9.81

        self.mppi_controller = MPPI_Numba(self.cfg)

        self.mppi_params = {
            'dt': self.cfg.dt,
            'x0': self.current_state,
            'xgoal': np.array([0, 0, 0.8, 0, 0, 0, -0.0, 0.0, -0.0, 0, 0, 0]),
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

        # MAVLink transmitter
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()

        # ------------------------------
        # LQR (optionally used)
        # ------------------------------
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix
        self.lqr_controller.desired_x = self.mppi_params['xgoal']

        # ------------------------------
        # MPC Setup
        # ------------------------------
        # # Trial 1:
        # self.mpc_params = {
        #     'inertia': self.inertia_flat,
        #     'mass': self.hex_mass,
        #     'horizon': 5,
        #     'gravity': 9.81,
        #     'max_force': 10.0,
        #     'max_torque': 1,
        #     'control_weight': 0.2,
        #     'tracking_weight_pos': 6,
        #     'tracking_weight_vel': 2,
        #     'tracking_weight_att': 0.5,
        #     'tracking_weight_ang_vel': 0.1,
        #     'smoothness_weight': 0.05,
        #     'lqr_weights' : np.array([1, 0.2, 0.05, 0.01, 2e-3, 2e-3]), # p, v, rpy, rate, f, m
        #     'dt': 0.2
        # }
        # Trial 2:
        self.mpc_params = {
            'inertia': self.inertia_flat,
            'mass': self.hex_mass,
            'horizon': 5,
            'gravity': 9.81,
            'max_force': 10.0,
            'max_torque': 1,
            'control_weight': 0.2,
            'tracking_weight_pos': 6,
            'tracking_weight_vel': 2,
            'tracking_weight_att': 0.5,
            'tracking_weight_ang_vel': 0.1,
            'smoothness_weight': 0.05,
            'lqr_weights' : np.array([1, 0.2, 0.05, 0.01, 2e-3, 2e-3]), # p, v, rpy, rate, f, m
            'dt': 0.2
        }
        
        self.mpc = OneStepMPC(self.mpc_params)
        self.tube_mpc = TubeMPC(self.mpc_params)

        # [CHANGED/ADDED] We will store the MPC target that MPPI provides:
        self.mpc_target = self.mppi_params['xgoal'].copy()

        # [CHANGED/ADDED] Frequencies
        self.mppi_rate_hz = 5.0
        self.mpc_rate_hz = 80.0
        # We will run our spin loop at 50 Hz, and only do MPPI logic once every 5 iterations.
        print("Initialization Complete.\n")

    def initialize_hexarotor_parameters(self):
        # self.hex_mass = 7.0
        self.hex_mass = 8.5 # intentional mismatch to test adaptive control
        self.gravity_compensation_scale = 1.0
        # self.inertia_flat = np.array([0.115125971, 0.116524229, 0.230387752])
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        # self.inertia_flat = np.array([0.03, 0.07, 0.10]) # intentional mismatch to test adaptive control
        self.inertia_matrix = np.diag(self.inertia_flat)

    # ------------------------------
    # ROS Callbacks
    # ------------------------------
    def activate_callback(self, data):
        self.activate = data.data

    def odometry_callback(self, data):
        self.odom = data
        pose = data.pose.pose
        twist = data.twist.twist

        # Update self.current_state
        disturbance = np.random.randn(6)
        self.current_state[:3] = [pose.position.x, pose.position.y, pose.position.z] #+ 0.05 * disturbance[:3]
        self.current_state[3:6] = [twist.linear.x, twist.linear.y, twist.linear.z]   

        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        euler = euler_from_quaternion(quaternion)
        self.current_state[6:9] = euler #+ 0.01 * disturbance[3:] + [0.01, 0.02, 0.03]
        self.current_state[9:] = [twist.angular.x, twist.angular.y, twist.angular.z]

        # Debug publish
        att_msg = Vector3Stamped()
        att_msg.header.stamp = data.header.stamp
        att_msg.vector.x = euler[0]
        att_msg.vector.y = euler[1]
        att_msg.vector.z = euler[2]
        self.att_debug_pub.publish(att_msg)

        # Publish MPPI's current xgoal every 10 steps just for debugging
        if (hasattr(self, 'mppi_controller') and hasattr(self, 'mpc_target')):
            target_msg = PoseStamped()
            target_msg.header.stamp = data.header.stamp
            xg = self.mppi_controller.params['xgoal']
            target_msg.pose.position.x = xg[0]
            target_msg.pose.position.y = xg[1]
            target_msg.pose.position.z = xg[2]
            self.target_debug_pub.publish(target_msg)

            target_mpc_msg = PoseStamped()
            target_mpc_msg.header.stamp = data.header.stamp
            xg_mpc = self.mpc_target.copy()
            target_mpc_msg.pose.position.x = xg_mpc[0]
            target_mpc_msg.pose.position.y = xg_mpc[1]
            target_mpc_msg.pose.position.z = xg_mpc[2]
            self.target_mpc_debug_pub.publish(target_mpc_msg)

    def target_callback(self, data):
        print("Target Received")
        # Update MPPI's target
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
        # Update LQR's target if desired
        self.lqr_controller.desired_x = self.mppi_controller.params['xgoal'].copy()
        # Set mpc target for debugging purposes
        self.mpc_target = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
            0, 0, 0,
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            0, 0, 0
        ])
        # print("mpc target = " + self.mpc_target)

    # ------------------------------
    # Publishing
    # ------------------------------
    def publish_cmd(self, control_inputs):
        self.control_inputs.header.stamp = rospy.Time.now()
        self.control_inputs.wrench.force.x = control_inputs[0]
        self.control_inputs.wrench.force.y = control_inputs[1]
        self.control_inputs.wrench.force.z = control_inputs[2]
        self.control_inputs.wrench.torque.x = control_inputs[3]
        self.control_inputs.wrench.torque.y = control_inputs[4]
        self.control_inputs.wrench.torque.z = control_inputs[5]

        # Attitude + thrust commands for the transmitter
        quat = [0.0, control_inputs[0], -control_inputs[1], -control_inputs[2]]
        angular_rates = [control_inputs[3], -control_inputs[4], -control_inputs[5]]
        thrust = -control_inputs[2]

        if self.activate:
            self.transmitter.send_attitude_control(angular_rates, thrust, quat)
            elapsed_time_pid = rospy.Time.now() - self.last_time_pid_pos_publish
            if (elapsed_time_pid.to_sec() > 0.5):
                self.publish_position_pid()

        self.control_pub.publish(self.control_inputs)

    def publish_position_pid(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        z = self.odom.pose.pose.position.z

        traj = FixedTrajectory()
        traj.type = "Point"

        # Just an example of publishing a reference for debugging
        att1 = KeyValue()
        att1.key = "frame_id"
        att1.value = "world"
        traj.attributes.append(att1)

        att2 = KeyValue()
        att2.key = "height"
        att2.value = str(z)
        traj.attributes.append(att2)

        att3 = KeyValue()
        att3.key = "max_acceleration"
        att3.value = str(0.4)
        traj.attributes.append(att3)

        att4 = KeyValue()
        att4.key = "velocity"
        att4.value = str(0.1)
        traj.attributes.append(att4)

        att5 = KeyValue()
        att5.key = "x"
        att5.value = str(x)
        traj.attributes.append(att5)

        att6 = KeyValue()
        att6.key = "y"
        att6.value = str(y)
        traj.attributes.append(att6)

        self.fixed_traj_pub.publish(traj)

    # ------------------------------
    # Dynamics Utilities
    # ------------------------------
    def compute_gravity_compensation(self):
        orientation_quat = [
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w
        ]
        gravity_vector_world = np.array([0, 0, -self.hex_mass * 9.81])
        rotation_matrix = Rotation.from_quat(orientation_quat).as_matrix()
        gravity_vector_body = rotation_matrix.T.dot(gravity_vector_world)
        return gravity_vector_body

    def dynamics_update(self, state, control_inputs, dt):
        """
        Simple RK4 integration for the hexarotor dynamics, using LQR's hex_dynamics
        just as an example. Adjust or replace with your own vehicle model if needed.
        """
        k1 = self.lqr_controller.hex_dynamics(state, control_inputs) * dt
        k2 = self.lqr_controller.hex_dynamics(state + k1 / 2, control_inputs) * dt
        k3 = self.lqr_controller.hex_dynamics(state + k2 / 2, control_inputs) * dt
        k4 = self.lqr_controller.hex_dynamics(state + k3, control_inputs) * dt
        next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state

    # ------------------------------
    # Normalization helpers
    # (These are just examples—tune to your system)
    # ------------------------------
    def normalize_control_inputs_mppi(self, ctrl, gravity_body):
        """
        Example scaling for MPPI outputs -> body rates + thrust.
        Adjust as needed for your vehicle.
        """
        ctrl[0] = ctrl[0] / 29.64 * 0.3
        ctrl[1] = ctrl[1] / 26.96 * 0.3
        ctrl[2] = ctrl[2] / 61.78 * 0.591
        ctrl[3:6] = ctrl[3:6] * 1
        return ctrl

    def normalize_control_inputs_mpc(self, ctrl):
        """
        Example scaling for MPC outputs -> body rates + thrust.
        Adjust as needed for your vehicle.
        """
        hover_thrust = 0.6567

        ctrl[0] = ctrl[0] * 0.515336334
        ctrl[1] = ctrl[1] * 0.515336334
        ctrl[2] = ctrl[2] *  hover_thrust / (self.hex_mass * 9.81) 
        ctrl[3:6] = ctrl[3:6] * 0.5
        return ctrl

    # ------------------------------
    # MPPI / MPC Combined 
    # ------------------------------
    def run_mppi(self):
        """
        Run one iteration of MPPI to update the entire control sequence.
        Then store the result in self.optimal_control_seq.
        """
        # Shift the previous solution and update with the new state
        self.mppi_controller.shift_and_update(self.current_state, self.optimal_control_seq, num_shifts=1)
        self.optimal_control_seq = self.mppi_controller.solve()

    def forward_simulate_for_mpc_target(self):
        """
        Forward simulate MPPI's first control (or first few) to find the state
        at the end of the MPC horizon, then set that as self.mpc_target.
        """
        # Example: Single-step forward for 0.3s (OneStepMPC has dt=0.3).
        # If you wanted multiple steps, you'd do a for-loop here.
        mppi_u = self.optimal_control_seq[0, :].copy()

        # Gravity compensation if you're not already factoring it in:
        gravity_vector_body = self.compute_gravity_compensation()
        if not GRAVITY:
            mppi_u[:3] -= gravity_vector_body


        # Forward-simulate from current_state:
        next_st = self.dynamics_update(self.current_state.copy(), mppi_u, self.mpc_params['dt'])
        next_st[6:] = np.zeros(6)

        final_target = np.array([self.mppi_params['xgoal'][0], self.mppi_params['xgoal'][1], self.mppi_params['xgoal'][2]])
    
        
        thresholds = np.array([0.05, 0.05, 0.05])
        for i in range(3):
            if abs(self.current_state[i] - final_target[i]) < thresholds[i]:
                self.current_state[i] = final_target[i].copy()
        # This next_st is our new MPC target
        self.mpc_target = next_st

    def run_mpc(self):
        """
        Solve the one-step MPC using:
         - current_state
         - mpc_target (set by MPPI)
         - MPPI's first-control as the initial guess
        """
        # We'll use the first MPPI control as an initial guess
        ctrl_guess_mppi = self.optimal_control_seq[0, :].copy()

        # For simplicity, just pass the raw guess in:
        u_mpc = self.mpc.compute_control(
            self.current_state.copy(),
            self.mpc_target.copy(),
            np.zeros(6),
            self.mpc_params['dt']
        )
        return u_mpc
    
    def run_tube_mpc(self):
        """
        Solve the tube MPC using:
         - current_state as x_real
         - mpc_target as x_nom
         - A hover-level nominal input for u_nom
        Returns the *real* control to be applied (u_real).
        """
        x_real = self.current_state.copy()
        x_nom  = self.mpc_target.copy()

        # For simplicity, pick a hover nominal input: Fz=mg, others=0
        u_hover = np.array([
            0.0, 0.0,
            self.hex_mass * 9.81,
            0.0, 0.0, 0.0
        ])

        # The tube MPC's compute_control method requires (x_real, x_nom, u_nom)
        u_tube = self.tube_mpc.compute_control(x_real, x_nom, u_hover)
        return u_tube

    # ------------------------------
    # Main spin: 50 Hz (MPC), MPPI at 10 Hz
    # ------------------------------
    def spin(self):
        print(f'Starting control loop. MPC at {self.mpc_rate_hz} Hz, MPPI at {self.mppi_rate_hz} Hz.\n')
        rate = rospy.Rate(self.mpc_rate_hz)  # 50 Hz
        iteration = 0

        while not rospy.is_shutdown():
            # 1) MPPI is run every (self.mpc_rate_hz // self.mppi_rate_hz) iterations
            if iteration % (self.mpc_rate_hz // self.mppi_rate_hz) == 0:
                self.run_mppi()
                self.forward_simulate_for_mpc_target()

            # 2) At every iteration (50 Hz), run MPC
            current_time = rospy.Time.now().to_sec()
            # mpc_ctrl = self.run_mpc()
            mpc_ctrl = self.run_tube_mpc()

            # 3) Normalize MPC control and filter
            mpc_ctrl_norm = self.normalize_control_inputs_mpc(mpc_ctrl.copy())
            # filtered = self.lpf.filter(mpc_ctrl_norm.copy())

            # 4) Publish
            self.publish_cmd(mpc_ctrl_norm)
            # self.publish_cmd(self.optimal_control_seq[])

            iteration += 1
            rate.sleep()



# ---------------------------------------------------------------------
# Local Test Simulation (Without Gazebo/ROS)
# ---------------------------------------------------------------------
def dynamics_update_sim(x, u, dt):
    """
    The dynamics update for the hexarotor simulation.
    Uses the provided equations.
    """
    I_xx = 0.115125971
    I_yy = 0.116524229
    I_zz = 0.230387752

    mass = 7.00
    g = 9.81

    x_next = x.copy()

    # Position updates
    x_next[0] += dt * x[3]
    x_next[1] += dt * x[4]
    x_next[2] += dt * x[5]

    # Linear acceleration updates
    x_next[3] += dt * ((1/mass) * u[0] - g * (np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) + np.sin(x[6]) * np.sin(x[8])))
    x_next[4] += dt * ((1/mass) * u[1] - g * (np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) - np.sin(x[6]) * np.cos(x[8])))
    x_next[5] += dt * ((1/mass) * u[2] - g * (np.cos(x[6]) * np.cos(x[7])))

    # Orientation (Euler angles) updates
    x_next[6] += dt * (x[9] + x[10] * (math.sin(x[6]) * math.tan(x[7])) + x[11] * (math.cos(x[6]) * math.tan(x[7])))
    x_next[7] += dt * (x[10] * math.cos(x[6]) - x[11] * math.sin(x[6]))
    x_next[8] += dt * (x[10] * math.sin(x[6]) / math.cos(x[7]) + x[11] * math.cos(x[6]) / math.cos(x[7]))

    # Angular acceleration updates
    x_next[9]  += dt * ((1 / I_xx) * (u[3] + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
    x_next[10] += dt * ((1 / I_yy) * (u[4] - I_xx * x[9] * x[11] + I_zz * x[9] * x[11]))
    x_next[11] += dt * ((1 / I_zz) * (u[5] + I_xx * x[9] * x[10] - I_yy * x[9] * x[10]))

    return x_next

def local_test():
    """
    Run a local simulation for 2000 steps using the MPPI and MPC controllers
    in the same way as in the main function. At each simulation step, the current
    state is updated using the custom dynamics_update_sim function and the control
    inputs computed via the MPPI and MPC logic. Finally, plot all 12 state variables versus time.
    """
    dt = 0.02  # simulation time step (50 Hz)
    n_steps = 2000

    # Create a controller instance (initializes MPPI and MPC as in the main function)
    controller = ControlHexarotor()

    # Initialize simulation state: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    x = np.zeros(12)
    x[2] = 0.8  # initial altitude

    # Initialize history arrays for states and control inputs
    state_history = np.zeros((n_steps + 1, 12))
    state_history[0, :] = x.copy()
    control_history = np.zeros((n_steps, 6))

    # Simulation loop
    for i in range(n_steps):
        # Update controller's internal state with the simulated state
        controller.current_state = x.copy()

        # Every 5 iterations (~10 Hz), run MPPI and update the MPC target
        # if i % 5 == 0:
        #     controller.run_mppi()
        #     controller.forward_simulate_for_mpc_target()

        # At every iteration (50 Hz), run MPC to compute a control command
        # controller.mpc_target = controller.mppi_params["xgoal"]
        u_mpc = controller.run_mpc()

        # Normalize the MPC control input as in the main loop
        u_mpc_norm = controller.normalize_control_inputs_mpc(u_mpc.copy())

        # (Optional) Apply low-pass filtering if desired:
        # u_mpc_norm = controller.lpf.filter(u_mpc_norm.copy())

        # Store the control input for later analysis
        control_history[i, :] = u_mpc_norm.copy()

        # Update the simulated state using the custom dynamics_update_sim function
        x = dynamics_update_sim(x, u_mpc_norm, dt/10)
        state_history[i + 1, :] = x.copy()

    # Plot each of the 12 state variables versus time
    import matplotlib.pyplot as plt
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    fig1, axs1 = plt.subplots(4, 3, figsize=(15, 10))
    axs1 = axs1.flatten()
    for j in range(12):
        axs1[j].plot(t, state_history[:, j])
        axs1[j].set_title(f'State x[{j}]')
        axs1[j].set_xlabel('Time (s)')
        axs1[j].set_ylabel(f'x[{j}]')
        axs1[j].grid(True)
    plt.tight_layout()

    # Plot each of the 6 control inputs (u_mpc) versus time
    t_u = np.linspace(0, n_steps * dt, n_steps)
    fig2, axs2 = plt.subplots(2, 3, figsize=(12, 8))
    axs2 = axs2.flatten()
    for k in range(6):
        axs2[k].plot(t_u, control_history[:, k])
        axs2[k].set_title(f'Control u[{k}]')
        axs2[k].set_xlabel('Time (s)')
        axs2[k].set_ylabel(f'u[{k}]')
        axs2[k].grid(True)
    plt.tight_layout()
    plt.show()

def main():
    controller = ControlHexarotor()
    controller.spin()

if __name__ == "__main__":
    # If the argument "local_test" is provided, run the local simulation.
    if len(sys.argv) > 1 and sys.argv[1] == "local_test":
        local_test()
    else:
        main()
