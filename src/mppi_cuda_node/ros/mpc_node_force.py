#!/usr/bin/env python
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
import mppi_cuda_node.cfg.MPCParamsConfig as MPCParamsConfig


class MPCControllerNode(object):
    def __init__(self):
        rospy.init_node('mpc_controller', anonymous=True)
        rospy.loginfo("Initializing MPC + 3D Force PID Controller Node ...")

        # ------------------------------
        #  1) State & MPC Target
        # ------------------------------
        self.current_state = np.zeros(12)  # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.mpc_target    = np.zeros(12)
        self.mpc_target_base    = np.zeros(12)
        self.mpc_target_base[2] = 0.8
        self.activate      = False

        # ---------------------------------
        #  2) Initialize Hexarotor & MPC
        # ---------------------------------
        self.initialize_hexarotor_parameters()
        self.initialize_mpc()

        # ---------------------------------
        #  3) 3D Force PID Setup
        # ---------------------------------
        # a) Force measurement & desired force in 3D
        self.current_force_meas = np.zeros(3)   # [Fx_meas, Fy_meas, Fz_meas]
        self.desired_force_3d   = np.array([0.0, 0.0, -0.9805])

        # b) PID gains (you can set separate gains per axis, but we'll do scalars here)
        self.Kp_force = 0.1
        self.Ki_force = 0.001
        self.Kd_force = 0.0001

        # c) Integrator & previous error (for each axis)
        self.force_integrator_3d  = np.zeros(3)
        self.force_error_prev_3d  = np.zeros(3)

        self.offset_x = 0.0
        self.offset_x_dot = 0.0

    
        # ---------------------------------
        #  4) Publishers & Subscribers
        # ---------------------------------
        self.control_pub   = rospy.Publisher('/mppi_debug/control_cmd', WrenchStamped, queue_size=10)
        self.att_debug_pub = rospy.Publisher('/mppi_debug/att_debug',   Vector3Stamped, queue_size=10)
        self.fixed_traj_pub= rospy.Publisher("/fixed_trajectory",       FixedTrajectory, queue_size=10)
        self.filtered_ft= rospy.Publisher("/ft_data_filtered",       WrenchStamped, queue_size=10)

        # MAVLink transmitter
        self.transmitter = MavlinkTransmitter()
        self.transmitter.master.wait_heartbeat()

        # Subscribers
        rospy.Subscriber('/odometry',      Odometry,    self.odometry_callback)
        rospy.Subscriber('/mpc/target',    PoseStamped, self.mpc_target_callback)
        rospy.Subscriber('/mppi/activate', Bool,        self.activate_callback)
        rospy.Subscriber('/ft_data', WrenchStamped, self.force_sensor_callback)
        rospy.Subscriber('/mpc/wrenchtarget', WrenchStamped, self.wrench_target_callback)

        self.mpc_rate_hz = 100.0
        self.last_time_pid_pos_publish = rospy.Time.now()

        rospy.loginfo("Initialization complete.")

    # ---------------------------------
    #  Hex & MPC Setup
    # ---------------------------------
    def initialize_hexarotor_parameters(self):
        self.hex_mass = 6.15  # kg
        self.inertia_flat = np.array([0.21, 0.21, 0.40])
        self.inertia_matrix = np.diag(self.inertia_flat)

    def initialize_mpc(self):
        self.mpc_params = {
            'inertia': self.inertia_flat,
            'mass': self.hex_mass,
            'horizon': 30,
            'gravity': 9.81,
            'max_force': 10.0,
            'max_torque': 1.0,
            'control_weight': 0.3,
            'tracking_weight_pos': 100,
            'tracking_weight_vel': 1,
            'tracking_weight_att': 150,
            'tracking_weight_ang_vel': 5,
            'terminal_weight': 5,
            'smoothness_weight': 0.05,
            'dt': 0.01
        }
        self.mpc = MPC(self.mpc_params)

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
        self.tube_mpc = TubeMPC(self.mpc_tube_params)

        # Optional LQR
        self.lqr_controller = LqrController()
        self.lqr_controller.m = self.hex_mass
        self.lqr_controller.J = self.inertia_matrix

    # ---------------------------------
    #  ROS Callbacks
    # ---------------------------------
    def activate_callback(self, data):
        self.activate = data.data

    def force_sensor_callback(self, msg):
        """
        Store the measured 3D force from WrenchStamped.
        Adjust if your sensor orientation is different.
        """
        # Update the current force measurement
        self.current_force_meas[0] = msg.wrench.force.x
        self.current_force_meas[1] = msg.wrench.force.y
        self.current_force_meas[2] = msg.wrench.force.z

        # Convert the incoming measurement into numpy arrays
        force = np.array([msg.wrench.force.x,
                        msg.wrench.force.y,
                        msg.wrench.force.z])
        torque = np.array([msg.wrench.torque.x,
                        msg.wrench.torque.y,
                        msg.wrench.torque.z])

        # Initialize buffers if they don't exist
        if not hasattr(self, '_filter_buffer_forces'):
            self._filter_buffer_forces = []        # Raw force measurements
            self._filter_buffer_torques = []         # Raw torque measurements
            self._moving_avg_buffer_forces = []      # Moving averages for force
            self._moving_avg_buffer_torques = []       # Moving averages for torque
            self._max_buffer_size = 10
            self._max_buffer_size_mv_avg = 30

        # Append the raw measurements to the buffers
        self._filter_buffer_forces.append(force)
        self._filter_buffer_torques.append(torque)

        # Keep raw measurement buffers within the maximum size
        if len(self._filter_buffer_forces) > self._max_buffer_size:
            self._filter_buffer_forces.pop(0)
            self._filter_buffer_torques.pop(0)

        # Compute moving average on the raw measurements
        moving_avg_force = np.mean(np.array(self._filter_buffer_forces), axis=0)
        moving_avg_torque = np.mean(np.array(self._filter_buffer_torques), axis=0)

        # Append the moving average results to their own buffers
        self._moving_avg_buffer_forces.append(moving_avg_force)
        self._moving_avg_buffer_torques.append(moving_avg_torque)

        # Ensure the moving average buffers also stay within the maximum size
        if len(self._moving_avg_buffer_forces) > self._max_buffer_size_mv_avg:
            self._moving_avg_buffer_forces.pop(0)
            self._moving_avg_buffer_torques.pop(0)

        # Compute the median of the moving average results (per component)
        median_force = np.median(np.array(self._moving_avg_buffer_forces), axis=0)
        # Optionally, compute median for torque:
        # median_torque = np.median(np.array(self._moving_avg_buffer_torques), axis=0)

        # Create and publish the filtered force message
        filtered_ft_msg = WrenchStamped()
        filtered_ft_msg.header = msg.header
        filtered_ft_msg.wrench.force.x = median_force[0]
        filtered_ft_msg.wrench.force.y = median_force[1]
        filtered_ft_msg.wrench.force.z = median_force[2]
        self.filtered_ft.publish(filtered_ft_msg)


    def wrench_target_callback(self, msg):
        """
        Process the incoming WrenchStamped message, apply a median filter
        using numpy.median to smooth the force and torque signals, and update
        the desired force and (optionally) torque.
        """
        # Convert the incoming measurement into numpy arrays
        force = np.array([msg.wrench.force.x,
                        msg.wrench.force.y,
                        msg.wrench.force.z])

        # Update the desired force
        self.desired_force_3d[0] = force[0]
        self.desired_force_3d[1] = force[1]
        self.desired_force_3d[2] = force[2]

    def odometry_callback(self, data):
        self.odom = data
        pose  = data.pose.pose
        twist = data.twist.twist

        # Update state
        self.current_state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        self.current_state[3:6] = [twist.linear.x, twist.linear.y,  twist.linear.z]

        # Euler angles
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        euler = euler_from_quaternion(quaternion)
        self.current_state[6:9] = euler
        self.current_state[9:]  = [twist.angular.x, twist.angular.y, twist.angular.z]

        # Debug
        att_msg = Vector3Stamped()
        att_msg.header.stamp = data.header.stamp
        att_msg.vector.x = euler[0]
        att_msg.vector.y = euler[1]
        att_msg.vector.z = euler[2]
        self.att_debug_pub.publish(att_msg)

    def mpc_target_callback(self, data):
        self.mpc_target_base[0] = data.pose.position.x
        self.mpc_target_base[1] = data.pose.position.y
        self.mpc_target_base[2] = data.pose.position.z
        self.mpc_target_base[6] = data.pose.orientation.x
        self.mpc_target_base[7] = data.pose.orientation.y
        self.mpc_target_base[8] = data.pose.orientation.z

    # ---------------------------------
    #  MPC
    # ---------------------------------
    def run_mpc(self):
        """
        Solve standard MPC for the one-step control.
        """
        u_mpc = self.mpc.compute_control(
            self.current_state.copy(),
            self.mpc_target.copy(),
            np.zeros(6),
            self.mpc_params['dt']
        )
        return u_mpc

    def run_mpc_tube(self):
        """
        Example if you want to use tube MPC instead.
        """
        x_real = self.current_state.copy()
        x_nom  = self.mpc_target.copy()
        u_hover = np.array([0.0, 0.0, self.hex_mass*9.81, 0.0, 0.0, 0.0])
        u_tube = self.tube_mpc.compute_control(x_real, x_nom, u_hover)
        return u_tube

    # ---------------------------------
    #  3D Force PID 
    # ---------------------------------
    def compute_force_correction_3d(self, dt):
        """
        A full 3D PID on (F_des - F_meas).
          e(t)   = F_des - F_meas
          de/dt  = (e - e_prev)/dt
          int_e += e*dt
          F_corr = Kp*e + Ki*int_e + Kd*(de/dt)
        """
        # print("current_force_meas: ", self.current_force_meas)
        # print("desired_force_3d: ", self.desired_force_3d)
        error_3d = self.desired_force_3d - self.current_force_meas
        derror_3d = (error_3d - self.force_error_prev_3d) / dt

        self.force_integrator_3d += error_3d * dt

        F_corr_3d = (self.Kp_force * error_3d
                     + self.Ki_force * self.force_integrator_3d
                     + self.Kd_force * derror_3d)

        # Update memory
        self.force_error_prev_3d = error_3d.copy()

        return F_corr_3d

    # ---------------------------------
    #  Command Publishing
    # ---------------------------------
    def normalize_control_inputs_mpc(self, ctrl):
        """
        Same approach you had before:
        scale the raw MPC outputs into actual thrust/torques.
        """
        hover_thrust = 0.62
        cos_phi   = np.cos(self.current_state[6])
        cos_theta = np.cos(self.current_state[7])
        sin_phi   = np.sin(self.current_state[6])
        sin_theta = np.sin(self.current_state[7])

        F_x_nom = self.hex_mass * 9.81 * sin_theta
        F_y_nom = -self.hex_mass * 9.81 * sin_phi * cos_theta
        hover_thrust_x = -hover_thrust * sin_theta * 2
        hover_thrust_y =  hover_thrust * sin_phi  * cos_theta * 1.8
        norm_factor_x = 0.02
        norm_factor_y = 0.02

        ctrl[0] = hover_thrust_x + norm_factor_x * (ctrl[0] - F_x_nom)
        ctrl[1] = hover_thrust_y + norm_factor_y * (ctrl[1] - F_y_nom)

        Fz_hover = self.hex_mass * 9.81 * cos_phi * cos_theta
        norm_factor_z = 0.05
        ctrl[2] = hover_thrust * cos_phi * cos_theta + norm_factor_z * (ctrl[2] - Fz_hover)
        ctrl[3:6] *= 0.24
        return ctrl

    def publish_cmd(self, control_inputs):
        cmd_msg = WrenchStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.wrench.force.x  = control_inputs[0]
        cmd_msg.wrench.force.y  = control_inputs[1]
        cmd_msg.wrench.force.z  = control_inputs[2]
        cmd_msg.wrench.torque.x = control_inputs[3]
        cmd_msg.wrench.torque.y = control_inputs[4]
        cmd_msg.wrench.torque.z = control_inputs[5]

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
        att1 = KeyValue(key="frame_id", value="world")
        att2 = KeyValue(key="height", value=str(z))
        att3 = KeyValue(key="max_acceleration", value=str(0.4))
        att4 = KeyValue(key="velocity", value=str(0.1))
        att5 = KeyValue(key="x", value=str(x))
        att6 = KeyValue(key="y", value=str(y))
        traj.attributes.extend([att1, att2, att3, att4, att5, att6])
        self.fixed_traj_pub.publish(traj)
        self.last_time_pid_pos_publish = rospy.Time.now()

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

    # ---------------------------------
    #  Main Loop
    # ---------------------------------
    def spin(self):
        rate = rospy.Rate(self.mpc_rate_hz)
        dt = 1.0 / self.mpc_rate_hz

        while not rospy.is_shutdown():
            
            self.mpc_target = self.mpc_target_base.copy()

            f_x_error = self.desired_force_3d[0] - self.current_force_meas[0]
            k_admittance = 10
            k_stiffness = 100
            c_damping = 160
    
            offset_accel = k_admittance * f_x_error - c_damping * self.offset_x_dot - k_stiffness * self.offset_x
            self.offset_x_dot += offset_accel * self.mpc_params['dt']
            self.offset_x += self.offset_x_dot * self.mpc_params['dt']
            self.mpc_target[0] += self.offset_x

            # 1) Run MPC
            u_mpc = self.run_mpc()

            # 2) Normalize the MPC output
            u_mpc_norm = self.normalize_control_inputs_mpc(u_mpc.copy())

            # 3) Compute the 3D force correction using PID
            force_correction_3d = self.compute_force_correction_3d(dt)

            # 4) Add the correction to [Fx, Fy, Fz] of the MPC output
            # u_mpc_norm[0] += force_correction_3d[0]
            # u_mpc_norm[1] += force_correction_3d[1]
            # u_mpc_norm[2] += force_correction_3d[2]

            # 5) Publish final command
            self.publish_cmd(u_mpc_norm)

            rate.sleep()


if __name__ == '__main__':
    try:
        node = MPCControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
