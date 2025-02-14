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
import casadi as cs
import torch


# --- MPC imports ---
from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC
from mppi_cuda_node.controllers.mpc.acados.acados_mpc_tube import TubeMPC
from mppi_cuda_node.controllers.lqr.lqr_controller import LqrController
from mppi_cuda_node.misc.mavlink.mavlink_transmitter import MavlinkTransmitter
from mppi_cuda_node.controllers.l1.l1_adaptive import L1AdaptiveController
from scipy.spatial.transform import Rotation
from mppi_cuda_node.misc.sim_noise_model.train_noise_model import NoiseNet

from dynamic_reconfigure.server import Server
# from mppi_cuda_node.cfg.MPCParamsConfig import MPCParamsConfig
import mppi_cuda_node.cfg.MPCParamsConfig as MPCParamsConfig



class MPCControllerNode(object):
    def __init__(self):
        rospy.init_node('mpc_controller', anonymous=True)
        rospy.loginfo("Initializing MPC Controller Node ...")

        self.current_state = np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.prev_state = np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.u_mpc = np.zeros(6)
        self.prev_u_mpc = np.zeros(6)
        self.u_total = np.zeros(6)
        self.prev_u_total = np.zeros(6)
        self.mpc_target = np.zeros(12)     # To be updated from the MPPI node
        self.mpc_target[2] = 0.8
        self.activate = False

        self.initialize_hexarotor_parameters()

        # ----- MPC Setup -----
        self.mpc_l1_params = {
            # MPC parameters:
            'inertia': self.inertia_flat,
            'mass': self.hex_mass,
            'horizon': 30,
            'gravity': 9.81,
            'max_force': 10.0,
            'max_torque': 1,
            'control_weight': 0.3,
            'tracking_weight_pos': 100,
            'tracking_weight_vel': 3,
            'tracking_weight_att': 100,
            'tracking_weight_ang_vel': 5,
            'terminal_weight': 0.1,
            'smoothness_weight': 0.05,
            'dt': 0.01,
            # L1 adaptive controller parameters:
            # 'l1_adaptation_gain': 0.0,
            # 'l1_filter_cutoff': 0.0000001
            'l1_adaptation_gain_pos_vertical':   0.0,#05,
            'l1_adaptation_gain_pos_horizontal': 0.0,#05,
            'l1_adaptation_gain_att':            0.0,#10,
            'l1_filter_cutoff': 25
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
        
        f_nominal = cs.Function('f_nominal', [self.mpc.model.x, self.mpc.model.u], [self.mpc.model.f_expl_expr]) # Create a CasADi function for the nominal dynamics for the L1 controller).
        self.l1_adaptive = L1AdaptiveController(self.mpc_l1_params, f_nominal)

        # LQR controller for auxiliary purposes (if needed)
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix

        # ----- Publishers -----
        self.control_pub = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.control_pub_umpc = rospy.Publisher('/mppi_debug/control_cmd_umpc', WrenchStamped, queue_size=10)
        self.control_pub_uadapt = rospy.Publisher('/mppi_debug/control_cmd_uadapt', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug', Vector3Stamped, queue_size=10)
        self.fixed_traj_pub = rospy.Publisher("/fixed_trajectory", FixedTrajectory, queue_size=10)

        # ----- MAVLink transmitter -----
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()

        # ----- Subscribers -----
        rospy.Subscriber('/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/mpc/target', PoseStamped, self.mpc_target_callback)
        rospy.Subscriber('/mppi/activate', Bool, self.activate_callback)

        self.mpc_rate_hz = 100.0  # Run MPC at 50 Hz
        self.last_time_pid_pos_publish = rospy.Time.now()

        self.sim = rospy.get_param("/use_sim_time", False)
        self.noise_model = None
        # Initialize the noise model.
        # self.noise_model = NoiseNet(input_dim=18, output_dim=12)
        self.noise_model = NoiseNet(input_dim=18, hidden_dim=64, num_layers=2, output_dim=12)
        model_path = "/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src/mppi_cuda_node/src/mppi_cuda_node/misc/sim_noise_model/noise_model_nn.pt"  # Ensure this file is in your working directory or provide full path
        self.noise_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.noise_model.to("cpu")
        self.noise_model.eval()

        # Set up dynamic reconfigure server for tuning MPC parameters
        # self.dyn_server = Server(MPCParamsConfig, self.dynamic_reconfigure_callback)

        rospy.loginfo("MPC Controller Node Initialization Complete.")

    def initialize_hexarotor_parameters(self):
        # Set your hexarotor parameters (tweak as needed)
        self.hex_mass = 7  # kg (example value)
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        # self.inertia_flat = np.array([0.71, 0.71, 0.90]) # mismatch
        self.inertia_matrix = np.diag(self.inertia_flat)

    def dynamic_reconfigure_callback(self, config, level):
        rospy.loginfo("Reconfigure Request:\nhorizon = %d\ndt = %.3f\nmax_force = %.2f\nmax_torque = %.2f\ncontrol_weight = %.2f\ntracking_weight_pos = %.2f\ntracking_weight_vel = %.2f\ntracking_weight_att = %.2f\ntracking_weight_ang_vel = %.2f\nsmoothness_weight = %.2f",
                      config['horizon'],config['dt'],config['max_force'],config['max_torque'],config['control_weight'],config['tracking_weight_pos'],config['tracking_weight_vel'],config['tracking_weight_att'],config['tracking_weight_ang_vel'],config['smoothness_weight'])
        # Update your MPC parameters here
        self.mpc_l1_params['horizon'] = config['horizon']
        self.mpc_l1_params['dt'] = config['dt']
        self.mpc_l1_params['max_force'] = config['max_force']
        self.mpc_l1_params['max_torque'] = config['max_torque']
        self.mpc_l1_params['control_weight'] = config['control_weight']
        self.mpc_l1_params['tracking_weight_pos'] = config['tracking_weight_pos']
        self.mpc_l1_params['tracking_weight_vel'] = config['tracking_weight_vel']
        self.mpc_l1_params['tracking_weight_att'] = config['tracking_weight_att']
        self.mpc_l1_params['tracking_weight_ang_vel'] = config['tracking_weight_ang_vel']
        self.mpc_l1_params['smoothness_weight'] = config['smoothness_weight']
        self.mpc.update_parameters(self.mpc_l1_params)
        return config


    def activate_callback(self, data):
        self.activate = data.data


    def odometry_callback(self, data):
        # Save previous state.
        self.prev_state = self.current_state.copy()
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

        # Angular velocities.
        self.current_state[9:] = [twist.angular.x, twist.angular.y, twist.angular.z]

        # ---- Use the noise model to predict the model mismatch (noise) ----
        # The noise network was trained with an 18D input: [state (12D); control (6D)]
        # Ensure that self.last_control is available (e.g., from your control loop)
        if self.sim:
            input_vec = np.concatenate([self.current_state, self.prev_u_total], axis=0)
            input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to('cpu')
            if self.noise_model is not None:
                with torch.no_grad():
                    noise_pred = self.noise_model(input_tensor)
            # Convert prediction to a 1D NumPy array (12D)
            noise_pred = noise_pred.cpu().numpy().flatten()

            self.current_state[0:3] += noise_pred[0:3]/10
            self.current_state[6:9] += noise_pred[3:6]/10



    def mpc_target_callback(self, data):
        """
        Callback for the target published by the MPPI node.
        We update our internal MPC target state accordingly.
        """
        self.mpc_target[0] = data.pose.position.x
        self.mpc_target[1] = data.pose.position.y
        self.mpc_target[2] = data.pose.position.z
        # For simplicity, we zero the remaining state elements.

        self.mpc_target[3:] = 0.0
        # self.mpc_target[8] = data.pose.orientation.z
        
    def normalize_control_inputs_mpc(self, ctrl):
        """
        Normalize/scaling for MPC outputs (body rates and thrust).
        Adjust these gains to suit your vehicle.
        """
        hover_thrust = 0.6567
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
            self.mpc_l1_params['dt']
        )
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

    def publish_umpc_uadapt_debug(self, u_mpc, u_adapt):
        cmd_msg = WrenchStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.wrench.force.x =  u_mpc[0].copy()
        cmd_msg.wrench.force.y =  u_mpc[1].copy()
        cmd_msg.wrench.force.z =  u_mpc[2].copy()
        cmd_msg.wrench.torque.x = u_mpc[3].copy()
        cmd_msg.wrench.torque.y = u_mpc[4].copy()
        cmd_msg.wrench.torque.z = u_mpc[5].copy()
        self.control_pub_umpc.publish(cmd_msg)
        cmd_msg2 = WrenchStamped()
        cmd_msg2.header.stamp = rospy.Time.now()
        cmd_msg2.wrench.force.x =  u_adapt[0].copy()
        cmd_msg2.wrench.force.y =  u_adapt[1].copy()
        cmd_msg2.wrench.force.z =  u_adapt[2].copy()
        cmd_msg2.wrench.torque.x = u_adapt[3].copy()
        cmd_msg2.wrench.torque.y = u_adapt[4].copy()
        cmd_msg2.wrench.torque.z = u_adapt[5].copy()
        self.control_pub_uadapt.publish(cmd_msg2)

    def spin(self):
        rate = rospy.Rate(self.mpc_rate_hz)
        while not rospy.is_shutdown():
            self.prev_u_mpc = self.u_mpc
            self.prev_u_total = self.u_total
            self.u_mpc = self.run_mpc()
            u_adapt = self.l1_adaptive.update(self.current_state.copy(), self.prev_state.copy(), self.prev_u_mpc.copy(), dt=(1/self.mpc_rate_hz))
            self.publish_umpc_uadapt_debug(self.u_mpc.copy(), u_adapt.copy())
            self.u_total = self.u_mpc + u_adapt
            u_total_norm = self.normalize_control_inputs_mpc(self.u_total.copy())
            self.publish_cmd(u_total_norm)
            rate.sleep()


if __name__ == '__main__':
    try:
        node = MPCControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
