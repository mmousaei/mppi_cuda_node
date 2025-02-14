#!/usr/bin/env python
import argparse
import os
import glob
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

import casadi as cs

# Uncomment if you need your MPC class
# from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC

import pysindy as ps

def get_active_period(bagfile_path):
    """
    Determines the active flight period in a bagfile.
    Active period starts at the first instance when /mppi/activate becomes True,
    and ends at the first instance (after activation) when /mppi/activate becomes False
    or when mavros/state reports mode "AUTO.LAND".
    """
    bag = rosbag.Bag(bagfile_path)
    bag_start = None
    t_begin = None
    t_end = None
    last_rel_time = 0.0
    for topic, msg, t in bag.read_messages(topics=["/mppi/activate", "mavros/state"]):
        t_sec = t.to_sec()
        if bag_start is None:
            bag_start = t_sec
        rel_t = t_sec - bag_start
        last_rel_time = rel_t
        if topic == "/mppi/activate":
            if t_begin is None and msg.data == True:
                t_begin = rel_t
            elif t_begin is not None and msg.data == False and t_end is None:
                t_end = rel_t
        elif topic == "mavros/state":
            if t_begin is not None and msg.mode == "AUTO.LAND" and t_end is None:
                t_end = rel_t
        if t_begin is not None and t_end is not None:
            break
    bag.close()
    if t_begin is None:
        return None, None
    if t_end is None:
        t_end = last_rel_time
    return t_begin, t_end

def load_data_from_bagfile(bagfile_path, state_topic, control_topic, t_begin, t_end):
    """
    Loads state and control messages from a bagfile within the time window [t_begin, t_end].
    Converts:
      - A state message (e.g. nav_msgs/Odometry) to a 12D vector.
      - A control message (e.g. WrenchStamped) to a 6D vector.
    """
    bag = rosbag.Bag(bagfile_path)
    t_state_list = []
    state_list = []
    t_control_list = []
    control_list = []
    bag_start = None

    for topic, msg, t in bag.read_messages(topics=[state_topic, control_topic]):
        t_sec = t.to_sec()
        if bag_start is None:
            bag_start = t_sec
        rel_t = t_sec - bag_start
        if rel_t < t_begin or rel_t > t_end:
            continue

        if topic == state_topic:
            state = np.zeros(12)
            # Position:
            state[0] = msg.pose.pose.position.x
            state[1] = msg.pose.pose.position.y
            state[2] = msg.pose.pose.position.z
            # Linear velocities:
            state[3] = msg.twist.twist.linear.x
            state[4] = msg.twist.twist.linear.y
            state[5] = msg.twist.twist.linear.z
            # Orientation: convert quaternion to Euler angles
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx*qx + qy*qy)
            state[6] = np.arctan2(sinr_cosp, cosr_cosp)
            sinp = 2 * (qw * qy - qz * qx)
            state[7] = np.arcsin(np.clip(sinp, -1, 1))
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
            state[8] = np.arctan2(siny_cosp, cosy_cosp)
            # Angular velocities:
            state[9]  = msg.twist.twist.angular.x
            state[10] = msg.twist.twist.angular.y
            state[11] = msg.twist.twist.angular.z

            t_state_list.append(rel_t)
            state_list.append(state)

        elif topic == control_topic:
            control = np.zeros(6)
            control[0] = msg.wrench.force.x
            control[1] = msg.wrench.force.y
            control[2] = msg.wrench.force.z
            control[3] = msg.wrench.torque.x
            control[4] = msg.wrench.torque.y
            control[5] = msg.wrench.torque.z

            t_control_list.append(rel_t)
            control_list.append(control)

    bag.close()
    return (np.array(t_state_list), np.array(state_list),
            np.array(t_control_list), np.array(control_list))

if __name__ == "__main__":
    # ======== Parameters ========
    # Replace these with your actual bagfile path and topic names.
    bagfile_path = '/home/dream_reaper/bags/noise_model_bags/2025-02-12-07-44-12.bag'
    state_topic = '/odometry'    # Example: state topic
    control_topic = '/mppi_debug/control_cmd'                # Example: control topic
    # ============================

    # Determine the active flight period.
    t_begin, t_end = get_active_period(bagfile_path)
    if t_begin is None:
        raise ValueError("No active flight period found in the bagfile.")

    # Load state and control data from the bagfile within the active period.
    t_state, X, t_control, U = load_data_from_bagfile(bagfile_path, state_topic, control_topic, t_begin, t_end)

    # --- Synchronize Control Data to the State Time Base ---
    # If state and control data have different timestamps, interpolate control data
    # so that we have a control input for every state sample.
    U_sync = np.zeros((len(t_state), U.shape[1]))
    for i in range(U.shape[1]):
        control_interp = interp1d(t_control, U[:, i], kind='linear', fill_value="extrapolate")
        U_sync[:, i] = control_interp(t_state)

    # --- (Optional) Smooth the State Data ---
    # Smoothing can help reduce noise prior to numerical differentiation.
    window_length = 21  # Must be an odd number; adjust as needed.
    polyorder = 3
    X_smooth = savgol_filter(X, window_length=window_length, polyorder=polyorder, axis=0)

    # ======== Set Up and Fit the SINDy Model ========
    # Use a finite-difference method to compute time derivatives.
    differentiation_method = ps.FiniteDifference(drop_endpoints=True)

    # Build candidate libraries:
    # A polynomial library captures nonlinearities.
    # A Fourier library captures periodic functions (useful for angular dynamics).
    poly_library = ps.PolynomialLibrary(degree=3)
    fourier_library = ps.FourierLibrary(n_frequencies=2)
    library = ps.GeneralizedLibrary(libraries=[poly_library, fourier_library])

    # Set up the sparse regression optimizer.
    optimizer = ps.STLSQ(threshold=0.1)

    # Initialize the SINDy model with control inputs.
    model = ps.SINDy(feature_library=library,
                     differentiation_method=differentiation_method,
                     optimizer=optimizer)

    # Fit the model to the smoothed state data with control inputs.
    model.fit(X_smooth, t=t_state, u=U_sync)

    # Print the discovered governing equations.
    # print("Learned Dynamics:")
    # model.print()

    # ======== Simulation and Comparison ========
    # Simulate the learned dynamics starting from the initial state.
    X_sim = model.simulate(X_smooth[0], t_state, u=U_sync)

    # Plot each state variable to compare measured vs. simulated data.
    n_states = X_smooth.shape[1]
    plt.figure(figsize=(15, 10))
    for i in range(n_states):
        plt.subplot(4, 3, i + 1)
        plt.plot(t_state, X_smooth[:, i], 'b', label='Measured')
        plt.plot(t_state, X_sim[:, i], 'r--', label='SINDy Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel(f'State {i+1}')
        plt.title(f'State {i+1}')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()
