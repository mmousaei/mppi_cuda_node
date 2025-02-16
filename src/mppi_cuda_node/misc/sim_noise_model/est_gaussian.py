#!/usr/bin/env python

import argparse
import os
import glob
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import casadi as cs

# Import your MPC class (ensure PYTHONPATH is set appropriately)
from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC

def get_active_period(bagfile_path):
    """
    Determines the active flight period in a bagfile.
    Active period starts at the first instance when /mppi/activate becomes True,
    and ends at the first instance (after activation) when /mppi/activate becomes False
    or when mavros/state reports mode "AUTO.LAND".
    
    Parameters:
      bagfile_path : path to the bagfile.
      
    Returns:
      (t_begin, t_end) : tuple of start and stop times (in seconds, relative to bag start).
                         If no activation is found, returns (None, None).
    """
    bag = rosbag.Bag(bagfile_path)
    bag_start = None
    t_begin = None
    t_end = None
    last_rel_time = 0.0
    for topic, msg, t in bag.read_messages(topics=["/mppi/activate", "/mavros/state", "/odometry"]):
        t_sec = t.to_sec()
        if topic == "/odometry" and bag_start is None:
            bag_start = t_sec
        rel_t = t_sec - bag_start
        last_rel_time = rel_t

        if topic == "/mppi/activate":
            # Assuming msg.data is a boolean.
            if t_begin is None and msg.data == True:
                t_begin = rel_t
            elif t_begin is not None and msg.data == False and t_end is None:
                t_end = rel_t
        elif topic == "/mavros/state":
            # Assuming msg.mode is a string.
            if t_begin is not None and msg.mode == "AUTO.LAND" and t_end is None:
                t_end = rel_t
        # If both are set, we can break early.
        if t_begin is not None and t_end is not None:
            break
    bag.close()
    if t_begin is None:
        print("beg")
        return None, None
    if t_end is None:
        print("end")
        return None, None
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

def main():
    parser = argparse.ArgumentParser(
        description="Learn a Gaussian Noise Model from the Residual Error (Predicted vs Real State)"
    )
    parser.add_argument("bag_dir", help="Directory containing bagfiles")
    parser.add_argument("--state_topic", default="/odometry", help="Topic for state messages")
    parser.add_argument("--control_topic", default="/mppi_debug/control_cmd", help="Topic for control messages")
    parser.add_argument("--smooth_window", type=int, default=51, help="Window length for Savitzky-Golay smoothing (must be odd)")
    parser.add_argument("--smooth_poly", type=int, default=3, help="Polynomial order for Savitzky-Golay smoothing")
    args = parser.parse_args()

    # Define MPC parameters.
    mpc_params = {
        'inertia': [0.115125971, 0.116524229, 0.230387752],
        'mass': 7.0,
        'gravity': 9.81,
        'horizon': 30,  # not used here
        'dt': 0.01,
        'max_force': 10.0,
        'max_torque': 1,
        'control_weight': 0.3,
        'tracking_weight_pos': 10,
        'tracking_weight_vel': 3,
        'tracking_weight_att': 30,
        'tracking_weight_ang_vel': 5,
        'terminal_weight': 1,
        'smoothness_weight': 0.05,
    }

    # Get list of bagfiles.
    bagfile_paths = glob.glob(os.path.join(args.bag_dir, "*.bag"))
    if len(bagfile_paths) == 0:
        print("No bagfiles found in the directory:", args.bag_dir)
        return

    all_states = []
    all_controls = []
    all_t_state = []
    all_t_control = []

    for bagfile in bagfile_paths:
        print("Processing bagfile:", bagfile)
        t_begin, t_end = get_active_period(bagfile)
        if t_begin is None or t_end is None:
            print("No active flight period found in", bagfile, "skipping...")
            continue
        t_state, states, t_control, controls = load_data_from_bagfile(
            bagfile, args.state_topic, args.control_topic, t_begin, t_end)
        if len(states) == 0 or len(controls) == 0:
            print("No state/control data found in", bagfile)
            continue
        print("Loaded {} state samples and {} control samples from {}".format(
            len(states), len(controls), bagfile))
        all_states.append(states)
        all_controls.append(controls)
        all_t_state.append(t_state)
        all_t_control.append(t_control)

    if len(all_states) == 0:
        print("No valid data loaded from any bagfiles.")
        return

    # Concatenate data from all bagfiles.
    states = np.concatenate(all_states, axis=0)    # shape (N, 12)
    controls = np.concatenate(all_controls, axis=0)  # shape (M, 6)
    t_state = np.concatenate(all_t_state, axis=0)    # shape (N,)
    t_control = np.concatenate(all_t_control, axis=0)  # shape (M,)

    print("Total state samples loaded:", len(states))
    print("Total control samples loaded:", len(controls))

    # Smooth the raw state data to reduce spikiness.
    smooth_states = savgol_filter(states, window_length=args.smooth_window, polyorder=args.smooth_poly, axis=0)

    # Interpolate control data to align with state timestamps.
    controls_aligned = np.zeros((len(t_state), controls.shape[1]))
    if len(t_control) > 1 and len(t_state) > 1:
        for ch in range(controls.shape[1]):
            interp_func = interp1d(t_control, controls[:, ch], kind='linear', fill_value="extrapolate")
            controls_aligned[:, ch] = interp_func(t_state)
    else:
        controls_aligned = controls

    # Instantiate the MPC to get access to its CasADi model.
    mpc = MPC(mpc_params)
    f_expl = mpc.model.f_expl_expr
    f_func = cs.Function("f_func", [mpc.model.x, mpc.model.u], [f_expl])

    # Compute residuals for each consecutive sample.
    residuals = []
    N_total = len(smooth_states)
    for i in range(N_total - 1):
        dt = t_state[i+1] - t_state[i]
        if dt <= 0:
            continue
        x_i = smooth_states[i]           # Full 12D state at time i
        x_next = smooth_states[i+1]        # Full 12D state at time i+1
        u_i = controls_aligned[i]         # Full 6D control at time i
        # Evaluate the predicted state derivative using the MPC model.
        f_val = np.array(f_func(x_i, u_i)).flatten()  # 12D derivative
        predicted_state = x_i + dt * f_val
        # Residual: difference between actual next state and predicted state.
        res = x_next - predicted_state
        residuals.append(res)

    residuals = np.array(residuals)  # shape (M, 12)
    print("Computed residuals shape:", residuals.shape)

    noise_mean = np.mean(residuals, axis=0)
    noise_cov = np.cov(residuals, rowvar=False)

    print("Estimated Gaussian Noise Mean:")
    print(noise_mean)
    print("Estimated Gaussian Noise Covariance:")
    print(noise_cov)

    np.savez("gaussian_noise.npz", mean=noise_mean, cov=noise_cov)
    print("Gaussian noise model saved to 'gaussian_noise.npz'")

    # Plot histograms for each dimension.
    num_dims = noise_mean.shape[0]
    for i in range(num_dims):
        plt.figure()
        plt.hist(residuals[:, i], bins=50)
        plt.title("Residual Histogram for Dimension {}".format(i))
        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
    plt.show()

    

if __name__ == "__main__":
    main()
