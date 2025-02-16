#!/usr/bin/env python
"""
estimate_params_from_acados_mpc.py

This script imports your MPC implementation from acados_mpc.py (which contains the OneStepMPC class)
and uses its dynamics model to estimate vehicle parameters:
  - Mass (from vertical dynamics)
  - Inertias (from angular dynamics)
  - 6D Normalization factors (for forces and torques)

It reads all the bagfiles in a specified folder. For each bagfile, the active flight period is defined as:
  - Start time: when the topic /mppi/activate first becomes True.
  - Stop time: when the topic /mppi/activate becomes False OR when the topic mavros/state has mode "AUTO.LAND"
    (whichever happens first).

The state and control data within this active period are extracted, combined across bagfiles, and then used for system identification.

**Note:** The controls stored in the bagfile are normalized via your
normalize_control_inputs_mpc() function. Therefore, they are de-normalized before estimation
using the inverse scaling:
  - For Fx, Fy: raw = norm / 0.515336334
  - For Fz: raw = norm * (nominal_mass*9.81/0.6567)
  - Torques remain unchanged.

Usage:
  python estimate_params_from_acados_mpc.py --bagfolder /path/to/bagfiles \
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
from scipy.optimize import least_squares

# Import your MPC implementation
from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC

# -------------------------------
# Helper function: Piecewise Gradient
# -------------------------------

def piecewise_gradient(x, t, gap_threshold=1.0):
    """
    Computes the gradient of x with respect to t piecewise, splitting the data when a time gap exceeds gap_threshold.
    
    Parameters:
      x : 1D numpy array.
      t : 1D numpy array of time stamps corresponding to x.
      gap_threshold : threshold for identifying discontinuities (default 1.0 second).
      
    Returns:
      grad : 1D numpy array of the same shape as x containing the gradient.
    """
    grad = np.zeros_like(x)
    # Identify indices where time gap is large
    dt = np.diff(t)
    # Find split indices where gap > gap_threshold
    split_indices = np.where(dt > gap_threshold)[0] + 1
    # Split indices into segments
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
    
    The normalization function in your code is:
      ctrl[0] = ctrl[0] * 0.515336334
      ctrl[1] = ctrl[1] * 0.515336334
      ctrl[2] = ctrl[2] * (hover_thrust/(hex_mass*9.81))   with hover_thrust = 0.6567
      ctrl[3:6] unchanged.
      
    Inverting these gives:
      raw_Fx = norm_Fx / 0.515336334
      raw_Fy = norm_Fy / 0.515336334
      raw_Fz = norm_Fz * (nominal_mass*9.81/0.6567)
      raw_torques = normalized torques.
      
    Parameters:
      ctrl_norm : numpy array of shape (N,6) with normalized controls.
      nominal_mass : Nominal mass used for normalization (e.g. 7.0 kg).
      
    Returns:
      ctrl_raw : numpy array of shape (N,6) with de-normalized controls.
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

# -------------------------------
# Data Loading
# -------------------------------

def load_data_from_bagfile(bagfile_path, state_topic, control_topic, t_begin, t_end):
    """
    Loads state and control messages from a bagfile within the time window [t_begin, t_end].
    Time is relative to the bagfile's start.
    
    Expected state message (e.g. nav_msgs/Odometry):
      [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
    Expected control message (e.g. WrenchStamped):
      [Fx, Fy, Fz, tau_x, tau_y, tau_z] (normalized)
    
    Parameters:
      bagfile_path : path to the bagfile.
      state_topic : topic for state messages.
      control_topic : topic for control messages.
      t_begin : start time (relative to bag start) to begin data extraction.
      t_end : end time (relative to bag start) to end data extraction.
      
    Returns:
      (t_state, states, t_control, controls) as numpy arrays.
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
            # Orientation: quaternion to Euler (roll, pitch, yaw)
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx*qx + qy*qy)
            state[6] = np.arctan2(sinr_cosp, cosr_cosp)
            sinp = 2 * (qw * qy - qz * qx)
            state[7] = np.arcsin(sinp) if np.abs(sinp) < 1 else np.sign(sinp)*math.pi/2
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy*qy + qz*qz)
            state[8] = np.arctan2(siny_cosp, cosy_cosp)
            # Angular velocity:
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
# Parameter Estimation Functions
# -------------------------------

def estimate_mass(t, states, controls_raw, g=9.81):
    """
    Estimate the mass using vertical dynamics, restricted to near-hover samples.
    
    In near hover (small roll and pitch, low vertical acceleration),
      m_i = F_z / (g * cos(phi) * cos(theta))
    and we take the median over such samples.
    
    Parameters:
      t : time array (in seconds)
      states : array of state vectors (each with 12 entries)
      controls_raw : array of raw controls (6 columns) with Fz in column 2
      g : gravity (default 9.81)
      
    Returns:
      m_est : estimated mass.
    """
    vz = states[:, 5]
    phi = states[:, 6]
    theta = states[:, 7]
    
    Fz = controls_raw[:, 2]
    
    near_hover = (np.abs(phi) < 0.1) & (np.abs(theta) < 0.1)
    
    if not np.any(near_hover):
        print("Warning: No near-hover samples found; cannot reliably estimate mass.")
        return None

    m_samples = Fz[near_hover] / (g * np.cos(phi[near_hover]) * np.cos(theta[near_hover]))
    m_est = np.median(m_samples)
    
    print("Median mass from near-hover samples:", m_est)
    return m_est

def residual_inertias(params, states, controls_raw, t):
    """
    Residual function for inertias estimation.
    
    For each time step, the model for roll dynamics is:
      p_dot = (1/I_xx) * (tau_x + (I_yy-I_zz)*q*r)
    and similarly for pitch and yaw.
    
    Rearranged, the residuals are:
      r1 = tau_x - I_xx * p_dot + (I_yy - I_zz)*q*r
      r2 = tau_y - I_yy * q_dot + (I_zz - I_xx)*p*r
      r3 = tau_z - I_zz * r_dot + (I_xx - I_yy)*p*q
    
    Parameters:
      params : array-like, [I_xx, I_yy, I_zz]
      states : state array with columns including angular velocities p, q, r at indices 9, 10, 11.
      controls_raw : raw control array with torques at indices 3, 4, 5.
      t : time array corresponding to the states.
      
    Returns:
      Concatenated residual vector.
    """
    I_xx, I_yy, I_zz = params
    p = states[:, 9]
    q = states[:, 10]
    r = states[:, 11]
    
    p_dot = piecewise_gradient(p, t)
    q_dot = piecewise_gradient(q, t)
    r_dot = piecewise_gradient(r, t)
    
    tau_x = controls_raw[:, 3]
    tau_y = controls_raw[:, 4]
    tau_z = controls_raw[:, 5]
    
    r1 = tau_x - I_xx * p_dot + (I_yy - I_zz) * q * r
    r2 = tau_y - I_yy * q_dot + (I_zz - I_xx) * p * r
    r3 = tau_z - I_zz * r_dot + (I_xx - I_yy) * p * q
    
    return np.concatenate((r1, r2, r3))

def estimate_inertias(t, states, controls_raw, I_guess):
    """
    Estimate the three inertias by minimizing the residuals from the angular dynamics.
    
    Parameters:
      t : time array corresponding to the states.
      states : state array.
      controls_raw : raw control array.
      I_guess : initial guess for [I_xx, I_yy, I_zz].
      
    Returns:
      Estimated inertias as an array.
    """
    lower_bounds = [1e-2, 1e-2, 1e-2]  # or some minimum values you deem appropriate
    upper_bounds = [np.inf, np.inf, np.inf]
    result = least_squares(lambda params: residual_inertias(params, states, controls_raw, t), I_guess,
                          bounds=(lower_bounds, upper_bounds))
    return result.x

def estimate_inertias_integral(t, states, controls_raw, I_guess, window_size=10.0):
    """
    Estimate the inertias Ixx, Iyy, Izz by integrating the rotational dynamics
    over fixed time windows. This avoids the noise introduced by numerical differentiation.
    
    The angular dynamics for roll, pitch, and yaw are:
      Roll:  p_dot = (1/Ixx)[tau_x + (Iyy - Izz)*q*r]
      Pitch: q_dot = (1/Iyy)[tau_y + (Izz - Ixx)*p*r]
      Yaw:   r_dot = (1/Izz)[tau_z + (Ixx - Iyy)*p*q]
    
    Integrating over a window [t0, t1] gives:
      ∫tau_x dt = Ixx * (p(t1)-p(t0)) - (Iyy - Izz) * ∫(q*r) dt
      ∫tau_y dt = Iyy * (q(t1)-q(t0)) - (Izz - Ixx) * ∫(p*r) dt
      ∫tau_z dt = Izz * (r(t1)-r(t0)) - (Ixx - Iyy) * ∫(p*q) dt
    
    We form these equations for several segments and solve for the inertias.
    
    Parameters:
      t           : 1D numpy array of time stamps.
      states      : 2D numpy array with at least 12 columns, where:
                    - p (roll rate) is column 9,
                    - q (pitch rate) is column 10,
                    - r (yaw rate) is column 11.
      controls_raw: 2D numpy array with 6 columns, where:
                    - tau_x is column 3,
                    - tau_y is column 4,
                    - tau_z is column 5.
      window_size : Duration (in seconds) of each integration window.
    
    Returns:
      Estimated inertias as a numpy array: [Ixx, Iyy, Izz].
      Returns None if not enough segments can be formed.
    """
    # Extract angular rates and torques.
    p   = states[:, 9]
    q   = states[:, 10]
    r   = states[:, 11]
    tau_x = controls_raw[:, 3]
    tau_y = controls_raw[:, 4]
    tau_z = controls_raw[:, 5]
    
    segments = []
    start_idx = 0
    n = len(t)
    
    # Create non-overlapping segments of length 'window_size'
    while start_idx < n:
        t0 = t[start_idx]
        end_time = t0 + window_size
        end_idx = start_idx
        while end_idx < n and t[end_idx] <= end_time:
            end_idx += 1
        
        # Skip segments with too few data points.
        if end_idx - start_idx < 2:
            break
        
        seg_t = t[start_idx:end_idx]
        seg_p = p[start_idx:end_idx]
        seg_q = q[start_idx:end_idx]
        seg_r = r[start_idx:end_idx]
        seg_tau_x = tau_x[start_idx:end_idx]
        seg_tau_y = tau_y[start_idx:end_idx]
        seg_tau_z = tau_z[start_idx:end_idx]
        
        # Compute differences and integrals over the segment.
        Delta_p = seg_p[-1] - seg_p[0]
        Delta_q = seg_q[-1] - seg_q[0]
        Delta_r = seg_r[-1] - seg_r[0]
        
        # Use trapezoidal integration.
        Q_qr = np.trapz(seg_q * seg_r, seg_t)
        Q_pr = np.trapz(seg_p * seg_r, seg_t)
        Q_pq = np.trapz(seg_p * seg_q, seg_t)
        
        T_x = np.trapz(seg_tau_x, seg_t)
        T_y = np.trapz(seg_tau_y, seg_t)
        T_z = np.trapz(seg_tau_z, seg_t)
        
        # Append a tuple for this segment.
        segments.append((Delta_p, Q_qr, T_x,
                         Delta_q, Q_pr, T_y,
                         Delta_r, Q_pq, T_z))
        
        start_idx = end_idx  # move to next segment
    
    if len(segments) < 1:
        print("Not enough segments for integral estimation")
        return None
    
    # Define the residual function for all segments.
    def residuals(params):
        Ixx, Iyy, Izz = params
        res = []
        for seg in segments:
            Delta_p, Q_qr, T_x, Delta_q, Q_pr, T_y, Delta_r, Q_pq, T_z = seg
            # Roll residual:
            res_roll = T_x - (Ixx * Delta_p - (Iyy - Izz) * Q_qr)
            # Pitch residual:
            res_pitch = T_y - (Iyy * Delta_q - (Izz - Ixx) * Q_pr)
            # Yaw residual:
            res_yaw = T_z - (Izz * Delta_r - (Ixx - Iyy) * Q_pq)
            res.extend([res_roll, res_pitch, res_yaw])
        return np.array(res)
    
    # Use a least-squares optimizer with positive bounds.
    lower_bounds = [1e-3, 1e-3, 1e-3]
    upper_bounds = [np.inf, np.inf, np.inf]
    result = least_squares(residuals, I_guess, bounds=(lower_bounds, upper_bounds))
    
    if not result.success:
        print("Integral-based inertia estimation did not converge.")
        return None
    return result.x

def estimate_normalization_factors(t, states, controls_norm, controls_raw, m_est, I_est, g=9.81):
    """
    Estimate the scaling factors for normalization of control inputs.
    
    Assumes that during near-hover conditions, the following approximations hold:
      Fx_raw = m_est * ax,
      Fy_raw = m_est * ay,
      Fz_raw = m_est * (az + g),
      tau_x_raw = I_est[0] * p_dot,
      tau_y_raw = I_est[1] * q_dot,
      tau_z_raw = I_est[2] * r_dot.
      
    Parameters:
      t : time array.
      states : state array.
      controls_norm : normalized control array.
      controls_raw : raw control array.
      m_est : estimated mass.
      I_est : estimated inertias [I_xx, I_yy, I_zz].
      g : gravity.
      
    Returns:
      Normalization factors for [Fx, Fy, Fz, tau_x, tau_y, tau_z].
    """
    phi = states[:, 6]
    theta = states[:, 7]
    vx = states[:, 3]
    vy = states[:, 4]
    vz = states[:, 5]
    near_hover = (np.abs(phi) < 0.1) & (np.abs(theta) < 0.1) & (np.abs(vx) < 0.1) & (np.abs(vy) < 0.1)
    if not np.any(near_hover):
        print("Warning: no near-hover segments found for normalization estimation.")
        near_hover = np.ones_like(vx, dtype=bool)
    
    ax = piecewise_gradient(vx, t)
    ay = piecewise_gradient(vy, t)
    vz_dot = piecewise_gradient(vz, t)
    
    p = states[:, 9]
    q = states[:, 10]
    r = states[:, 11]
    p_dot = piecewise_gradient(p, t)
    q_dot = piecewise_gradient(q, t)
    r_dot = piecewise_gradient(r, t)
    
    Fx_est = m_est * ax
    Fy_est = m_est * ay
    Fz_est = m_est * (vz_dot + g)
    
    tau_x_est = I_est[0] * p_dot
    tau_y_est = I_est[1] * q_dot
    tau_z_est = I_est[2] * r_dot
    
    def safe_ratio(norm, est):
        valid = (np.abs(est) > 1e-3) & near_hover
        if np.sum(valid) < 1:
            return 1.0
        return np.mean(norm[valid] / est[valid])
    
    s_Fx = safe_ratio(controls_norm[:, 0], Fx_est)
    s_Fy = safe_ratio(controls_norm[:, 1], Fy_est)
    s_Fz = safe_ratio(controls_norm[:, 2], Fz_est)
    s_tau_x = safe_ratio(controls_norm[:, 3], tau_x_est)
    s_tau_y = safe_ratio(controls_norm[:, 4], tau_y_est)
    s_tau_z = safe_ratio(controls_norm[:, 5], tau_z_est)
    
    return np.array([s_Fx, s_Fy, s_Fz, s_tau_x, s_tau_y, s_tau_z])

# -------------------------------
# Main Routine
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate mass, inertias, and 6D normalization factors using acados MPC dynamics model from flight data in bagfiles.")
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
        # The file was saved as a dictionary, so we use allow_pickle=True and then .item()
        data = np.load(data_file, allow_pickle=True).item()
        t_all = data["t_all"]
        states_all = data["states_all"]
        controls_all_norm = data["controls_all_norm"]
    else:
        print("No processed data file found. Processing bagfiles...")
        all_t = []
        all_states = []
        all_controls_norm = []  # these are normalized controls as recorded

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
            Fx_interp = interp1d(t_control, controls[:,0], kind='linear', fill_value="extrapolate")
            Fy_interp = interp1d(t_control, controls[:,1], kind='linear', fill_value="extrapolate")
            Fz_interp = interp1d(t_control, controls[:,2], kind='linear', fill_value="extrapolate")
            tau_x_interp = interp1d(t_control, controls[:,3], kind='linear', fill_value="extrapolate")
            tau_y_interp = interp1d(t_control, controls[:,4], kind='linear', fill_value="extrapolate")
            tau_z_interp = interp1d(t_control, controls[:,5], kind='linear', fill_value="extrapolate")
            controls_interp_norm = np.column_stack((Fx_interp(t_state),
                                                     Fy_interp(t_state),
                                                     Fz_interp(t_state),
                                                     tau_x_interp(t_state),
                                                     tau_y_interp(t_state),
                                                     tau_z_interp(t_state)))
            # Adjust time to be continuous across bagfiles.
            t_state_adjusted = t_state + global_time_offset
            if len(t_state_adjusted) > 0:
                global_time_offset = t_state_adjusted[-1] + 0.1  # add a small gap
            
            all_t.append(t_state_adjusted)
            all_states.append(states)
            all_controls_norm.append(controls_interp_norm)

        if not all_t:
            print("No valid data loaded. Exiting.")
            return

        # Concatenate and sort data by time.
        t_all = np.concatenate(all_t)
        states_all = np.concatenate(all_states)
        controls_all_norm = np.concatenate(all_controls_norm)
        sort_idx = np.argsort(t_all)
        t_all = t_all[sort_idx]
        states_all = states_all[sort_idx]
        controls_all_norm = controls_all_norm[sort_idx]

        # Save the processed data for future use.
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

    # Estimate mass using vertical dynamics.
    m_est = estimate_mass(t_all, states_all, controls_all_raw, g=params["gravity"])
    if m_est is None:
        print("Mass estimation failed. Exiting.")
        return
    print("Estimated mass: {:.3f} kg".format(m_est))

    # Estimate inertias using angular dynamics.
    I_guess = np.array(params["inertia"])  # initial guess from nominal parameters
    # I_est = estimate_inertias(t_all, states_all, controls_all_raw, I_guess)
    I_est = estimate_inertias_integral(t_all, states_all, controls_all_raw, I_guess, 10)
    print("Estimated inertias: I_xx={:.5f}, I_yy={:.5f}, I_zz={:.5f}".format(*I_est))

    # Estimate normalization factors.
    norm_factors = estimate_normalization_factors(t_all, states_all, controls_all_norm, controls_all_raw, m_est, I_est, g=params["gravity"])
    channels = ["Fx", "Fy", "Fz", "tau_x", "tau_y", "tau_z"]
    for ch, sf in zip(channels, norm_factors):
        print("Estimated normalization factor for {}: {:.5f}".format(ch, sf))

    # (Optional) Plot some results.
    plt.figure()
    plt.plot(t_all, states_all[:,5], label="vz")
    plt.xlabel("Time [s]")
    plt.ylabel("Vertical velocity [m/s]")
    plt.title("Vertical velocity vs. time")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
