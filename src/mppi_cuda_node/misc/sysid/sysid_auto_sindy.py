#!/usr/bin/env python
"""
estimate_params_from_acados_mpc_sindy.py

This script imports your MPC implementation from acados_mpc.py (which contains the OneStepMPC class)
and uses its dynamics model to estimate vehicle parameters:
  - Mass (from vertical dynamics)
  - Inertias (from angular dynamics)
  - 6D Normalization factors (for forces and torques)

It reads all the bagfiles in a specified folder. For each bagfile, the active flight period is defined as:
  - Start time: when the topic /mppi/activate first becomes True.
  - Stop time: when the topic /mppi/activate becomes False OR when the topic mavros/state has mode "AUTO.LAND"
    (whichever happens first).

The state and control data within this active period are extracted, combined across bagfiles, and then used for parameter estimation.

**Note:** The controls stored in the bagfile are normalized via your
normalize_control_inputs_mpc() function. Therefore, they are de-normalized before estimation
using the inverse scaling:
  - For Fx, Fy: raw = norm / 0.515336334
  - For Fz: raw = norm * (nominal_mass*9.81/0.6567)
  - Torques remain unchanged.

Usage:
  python estimate_params_from_acados_mpc_sindy.py --bagfolder /path/to/bagfiles \
         --state_topic /odometry --control_topic /mppi_debug/control_cmd
"""

import argparse
import os
import glob
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import PySINDy for parameter estimation
import pysindy as ps

# Import your MPC implementation
from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC

# -------------------------------
# Helper Function: Piecewise Gradient
# -------------------------------
def piecewise_gradient(x, t, gap_threshold=1.0):
    """
    Computes the gradient of x with respect to t piecewise, splitting the data when a time gap exceeds gap_threshold.
    """
    grad = np.zeros_like(x)
    dt = np.diff(t)
    split_indices = np.where(dt > gap_threshold)[0] + 1
    indices = np.arange(len(x))
    segments = np.split(indices, split_indices)
    for seg in segments:
        if len(seg) == 1:
            grad[seg[0]] = 0.0
        else:
            grad[seg] = np.gradient(x[seg], t[seg])
    return grad

# -------------------------------
# De-normalization Function
# -------------------------------
def denormalize_controls(ctrl_norm, nominal_mass):
    """
    Given normalized controls (as stored in the bagfile) based on your
    normalize_control_inputs_mpc() function, compute the corresponding raw controls.
    Inverting:
      raw_Fx = norm_Fx / 0.515336334
      raw_Fy = norm_Fy / 0.515336334
      raw_Fz = norm_Fz * (nominal_mass*9.81/0.6567)
      raw_torques = normalized torques.
    """
    hover_thrust = 0.6567
    scaling_factor_xy = 0.515336334
    Fx_raw = ctrl_norm[:, 0] / scaling_factor_xy
    Fy_raw = ctrl_norm[:, 1] / scaling_factor_xy
    Fz_raw = ctrl_norm[:, 2] * (nominal_mass * 9.81 / hover_thrust)
    tau_x_raw = ctrl_norm[:, 3]
    tau_y_raw = ctrl_norm[:, 4]
    tau_z_raw = ctrl_norm[:, 5]
    ctrl_raw = np.column_stack((Fx_raw, Fy_raw, Fz_raw, tau_x_raw, tau_y_raw, tau_z_raw))
    return ctrl_raw

# -------------------------------
# Active Period Extraction
# -------------------------------
def get_active_period(bagfile_path):
    """
    Determines the active flight period in a bagfile.
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
            if t_begin is None and msg.data == True:
                t_begin = rel_t
            elif t_begin is not None and msg.data == False and t_end is None:
                t_end = rel_t
        elif topic == "/mavros/state":
            if t_begin is not None and msg.mode == "AUTO.LAND" and t_end is None:
                t_end = rel_t
        if t_begin is not None and t_end is not None:
            break
    bag.close()
    if t_begin is None or t_end is None:
        return None, None
    return t_begin, t_end

# -------------------------------
# Data Loading
# -------------------------------
def load_data_from_bagfile(bagfile_path, state_topic, control_topic, t_begin, t_end):
    """
    Loads state and control messages from a bagfile within the time window [t_begin, t_end].
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
            state[0] = msg.pose.pose.position.x
            state[1] = msg.pose.pose.position.y
            state[2] = msg.pose.pose.position.z
            state[3] = msg.twist.twist.linear.x
            state[4] = msg.twist.twist.linear.y
            state[5] = msg.twist.twist.linear.z
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

# -------------------------------
# SINDy–Based Parameter Estimation Functions
# -------------------------------
def estimate_mass_sindy(t, states, controls_raw, g=9.81):
    """
    Estimate mass from vertical dynamics using SINDy.
    
    Our model:
       vz_dot + g*cos(phi)*cos(theta) = Fz_raw / m
    Using an identity library on Fz_raw, SINDy learns alpha = 1/m.
    """
    dt = np.median(np.diff(t))
    vz = states[:, 5]
    # Compute derivative using numpy.gradient (or another method)
    vz_dot = np.gradient(vz, dt)
    phi = states[:, 6]
    theta = states[:, 7]
    R = vz_dot + g * np.cos(phi) * np.cos(theta)  # residual
    
    Fz = controls_raw[:, 2].reshape(-1, 1)  # raw vertical force
    
    # Build SINDy model with an identity library (only candidate: Fz)
    library = ps.IdentityLibrary()  # removed n_input_features argument
    optimizer = ps.STLSQ(threshold=1e-12, alpha=0)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    # Regress R against Fz.
    model.fit(R.reshape(-1, 1), u=Fz, t=dt)
    coef = model.coefficients()  # expected shape: (1,1)
    alpha = coef[0, 0]
    m_est = 1 / alpha if alpha != 0 else None
    print("Estimated mass using SINDy: {:.3f} kg".format(m_est))
    return m_est

def estimate_inertias_sindy(t, states, controls_raw):
    """
    Estimate inertias using angular dynamics with SINDy.
    
    For roll dynamics, we assume:
       p_dot = a1*tau_x_raw + a2*(q*r)
    where a1 = 1/Ixx and a2 = (Iyy - Izz)/Ixx.
    Similar regressions are performed for pitch and yaw.
    """
    dt = np.median(np.diff(t))
    p = states[:, 9]
    q = states[:, 10]
    r = states[:, 11]
    p_dot = np.gradient(p, dt)
    q_dot = np.gradient(q, dt)
    r_dot = np.gradient(r, dt)
    
    # Roll dynamics:
    tau_x = controls_raw[:, 3].reshape(-1, 1)
    qr = (q * r).reshape(-1, 1)
    X_roll = np.hstack((tau_x, qr))
    library = ps.IdentityLibrary()  # removed n_input_features argument
    optimizer = ps.STLSQ(threshold=1e-12, alpha=0)
    model_roll = ps.SINDy(feature_library=library, optimizer=optimizer)
    model_roll.fit(p_dot.reshape(-1, 1), u=X_roll, t=dt)
    coef_roll = model_roll.coefficients()  # shape (1,2)
    a1 = coef_roll[0, 0]
    Ixx_est = 1 / a1 if a1 != 0 else None

    # Pitch dynamics:
    tau_y = controls_raw[:, 4].reshape(-1, 1)
    pr = (p * r).reshape(-1, 1)
    X_pitch = np.hstack((tau_y, pr))
    model_pitch = ps.SINDy(feature_library=library, optimizer=optimizer)
    model_pitch.fit(q_dot.reshape(-1, 1), u=X_pitch, t=dt)
    coef_pitch = model_pitch.coefficients()
    b1 = coef_pitch[0, 0]
    Iyy_est = 1 / b1 if b1 != 0 else None

    # Yaw dynamics:
    tau_z = controls_raw[:, 5].reshape(-1, 1)
    pq = (p * q).reshape(-1, 1)
    X_yaw = np.hstack((tau_z, pq))
    model_yaw = ps.SINDy(feature_library=library, optimizer=optimizer)
    model_yaw.fit(r_dot.reshape(-1, 1), u=X_yaw, t=dt)
    coef_yaw = model_yaw.coefficients()
    c1 = coef_yaw[0, 0]
    Izz_est = 1 / c1 if c1 != 0 else None

    print("Estimated inertias using SINDy:")
    print("Ixx: {:.5f}, Iyy: {:.5f}, Izz: {:.5f}".format(Ixx_est, Iyy_est, Izz_est))
    return np.array([Ixx_est, Iyy_est, Izz_est])

def estimate_normalization_factors_sindy(states, controls_norm, controls_raw):
    """
    Estimate static normalization factors using SINDy.
    
    For each channel we assume a linear static mapping:
         raw = s * norm.
    Using an identity library, the learned coefficient s is returned.
    """
    norm_factors = []
    library = ps.IdentityLibrary()  # removed n_input_features argument
    optimizer = ps.STLSQ(threshold=1e-12, alpha=0)
    dt = 1.0  # dummy time for static regression
    for i in range(6):
        norm = controls_norm[:, i].reshape(-1, 1)
        raw = controls_raw[:, i].reshape(-1, 1)
        model = ps.SINDy(feature_library=library, optimizer=optimizer)
        # Regress raw against norm.
        model.fit(raw, u=norm, t=dt)
        coef = model.coefficients()
        s = coef[0, 0]
        norm_factors.append(s)
        channels = ["Fx", "Fy", "Fz", "tau_x", "tau_y", "tau_z"]
        print("Estimated normalization factor for {}: {:.5f}".format(channels[i], s))
    return np.array(norm_factors)


# -------------------------------
# Main Routine
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Estimate mass, inertias, and 6D normalization factors using acados MPC dynamics model (SINDy version) from flight data in bagfiles.")
    parser.add_argument("--bagfolder", required=True,
                        help="Folder containing bagfiles")
    parser.add_argument("--state_topic", default="/odometry", help="State topic (default: /odometry)")
    parser.add_argument("--control_topic", default="/mppi_debug/control_cmd", help="Control topic (default: /mppi_debug/control_cmd)")
    args = parser.parse_args()

    bagfolder = args.bagfolder
    bag_files = sorted(glob.glob(os.path.join(bagfolder, "*.bag")))
    if not bag_files:
        print("No bagfiles found in folder:", bagfolder)
        return

    data_file = os.path.join(bagfolder, "processed_data.npy")
    
    if os.path.exists(data_file):
        print("Processed data file found. Loading data from", data_file)
        data = np.load(data_file, allow_pickle=True).item()
        t_all = data["t_all"]
        states_all = data["states_all"]
        controls_all_norm = data["controls_all_norm"]
    else:
        print("No processed data file found. Processing bagfiles...")
        all_t = []
        all_states = []
        all_controls_norm = []
        global_time_offset = 0.0

        for bag_path in bag_files:
            print("Processing bagfile:", bag_path)
            active_period = get_active_period(bag_path)
            if active_period[0] is None:
                print("  No active period found in bagfile; skipping.")
                continue
            t_begin, t_end = active_period
            print("  Active period: t_begin = {:.2f} s, t_end = {:.2f} s".format(t_begin, t_end))
            t_state, states, t_control, controls = load_data_from_bagfile(
                bag_path, args.state_topic, args.control_topic, t_begin, t_end)
            if len(t_state) < 2 or len(t_control) < 2:
                print("  Not enough data in active period for bagfile:", bag_path)
                continue
            # Interpolate normalized controls onto state timestamps.
            Fx_interp = interp1d(t_control, controls[:, 0], kind='linear', fill_value="extrapolate")
            Fy_interp = interp1d(t_control, controls[:, 1], kind='linear', fill_value="extrapolate")
            Fz_interp = interp1d(t_control, controls[:, 2], kind='linear', fill_value="extrapolate")
            tau_x_interp = interp1d(t_control, controls[:, 3], kind='linear', fill_value="extrapolate")
            tau_y_interp = interp1d(t_control, controls[:, 4], kind='linear', fill_value="extrapolate")
            tau_z_interp = interp1d(t_control, controls[:, 5], kind='linear', fill_value="extrapolate")
            controls_interp_norm = np.column_stack((Fx_interp(t_state),
                                                     Fy_interp(t_state),
                                                     Fz_interp(t_state),
                                                     tau_x_interp(t_state),
                                                     tau_y_interp(t_state),
                                                     tau_z_interp(t_state)))
            t_state_adjusted = t_state + global_time_offset
            if len(t_state_adjusted) > 0:
                global_time_offset = t_state_adjusted[-1] + 0.1
            all_t.append(t_state_adjusted)
            all_states.append(states)
            all_controls_norm.append(controls_interp_norm)

        if not all_t:
            print("No valid data loaded. Exiting.")
            return

        t_all = np.concatenate(all_t)
        states_all = np.concatenate(all_states)
        controls_all_norm = np.concatenate(all_controls_norm)
        sort_idx = np.argsort(t_all)
        t_all = t_all[sort_idx]
        states_all = states_all[sort_idx]
        controls_all_norm = controls_all_norm[sort_idx]

        data = {"t_all": t_all, "states_all": states_all, "controls_all_norm": controls_all_norm}
        np.save(data_file, data)
        print("Processed data saved to", data_file)

    # Create an instance of your MPC to access nominal parameters.
    params = {
        'inertia': [0.115125971, 0.116524229, 0.230387752],
        'horizon': 30,
        'mass': 7.00,  # nominal mass used for normalization
        'gravity': 9.81,
        'max_force': 20.0,
        'max_torque': 0.05,
        'control_weight': 0.005,
        'tracking_weight_pos': 10,
        'tracking_weight_vel': 3,
        'tracking_weight_att': 80,
        'tracking_weight_ang_vel': 50,
        'terminal_weight': 1,
        'smoothness_weight': 0.01,
        'dt': 0.3
    }
    mpc = MPC(params)

    # De-normalize the controls using the nominal mass (7.00 kg)
    controls_all_raw = denormalize_controls(controls_all_norm, nominal_mass=params["mass"])

    # --- SINDy–Based Estimation ---
    m_est = estimate_mass_sindy(t_all, states_all, controls_all_raw, g=params["gravity"])
    if m_est is None:
        print("Mass estimation failed. Exiting.")
        return
    print("Estimated mass: {:.3f} kg".format(m_est))

    I_est = estimate_inertias_sindy(t_all, states_all, controls_all_raw)
    print("Estimated inertias: I_xx={:.5f}, I_yy={:.5f}, I_zz={:.5f}".format(*I_est))

    norm_factors = estimate_normalization_factors_sindy(states_all, controls_all_norm, controls_all_raw)
    channels = ["Fx", "Fy", "Fz", "tau_x", "tau_y", "tau_z"]
    for ch, sf in zip(channels, norm_factors):
        print("Estimated normalization factor for {}: {:.5f}".format(ch, sf))

    # (Optional) Plot a sample: vertical velocity vs. time.
    plt.figure()
    plt.plot(t_all, states_all[:, 5], label="vz")
    plt.xlabel("Time [s]")
    plt.ylabel("Vertical velocity [m/s]")
    plt.title("Vertical velocity vs. time")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
