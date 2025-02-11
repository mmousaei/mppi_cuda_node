#!/usr/bin/env python
"""
estimate_params_from_acados_mpc.py

This script imports your MPC implementation from acados_mpc.py (which contains the OneStepMPC class)
and uses its dynamics model to estimate vehicle parameters:
  - Mass (from vertical dynamics)
  - Inertias (from angular dynamics)
  - 6D Normalization factors (for forces and torques)

**Note:** The controls stored in the bagfile are normalized via your
normalize_control_inputs_mpc() function. Therefore, they are de-normalized before estimation
using the inverse scaling:
  - For Fx, Fy: raw = norm / 0.515336334
  - For Fz: raw = norm * (nominal_mass*9.81/0.6567)
  - Torques remain unchanged.

Usage:
  python estimate_params_from_acados_mpc.py --bagfiles data1.bag,0,100 data2.bag,120,200 \
         --state_topic /odometry --control_topic /mppi_debug/control_cmd
"""

import argparse
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

# Import your MPC implementation
from acados_mpc import OneStepMPC

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
    # ctrl_norm is assumed to be 2D: (N,6)
    Fx_raw = ctrl_norm[:, 0] / scaling_factor_xy
    Fy_raw = ctrl_norm[:, 1] / scaling_factor_xy
    Fz_raw = ctrl_norm[:, 2] * (nominal_mass * 9.81 / hover_thrust)
    tau_x_raw = ctrl_norm[:, 3]
    tau_y_raw = ctrl_norm[:, 4]
    tau_z_raw = ctrl_norm[:, 5]
    ctrl_raw = np.column_stack((Fx_raw, Fy_raw, Fz_raw, tau_x_raw, tau_y_raw, tau_z_raw))
    return ctrl_raw

# -------------------------------
# Data Loading
# -------------------------------

def load_data_from_bagfile(bagfile_path, state_topic, control_topic, t_begin, t_end):
    """
    Loads state and control messages from a bagfile.
    Time is made relative using the first message timestamp.
    Only messages with relative time in [t_begin, t_end] are used.
    
    Expected state message (e.g. nav_msgs/Odometry):
      [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
    Expected control message (e.g. WrenchStamped):
      [Fx, Fy, Fz, tau_x, tau_y, tau_z] (normalized)
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
    Estimate the mass using vertical dynamics, but restrict the estimation to near-hover samples.
    
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
    # Extract needed state components
    vz = states[:, 5]
    phi = states[:, 6]
    theta = states[:, 7]
    
    # Extract vertical thrust (Fz)
    Fz = controls_raw[:, 2]
    
    # Use the near-hover region: small roll and pitch
    near_hover = (np.abs(phi) < 0.1) & (np.abs(theta) < 0.1)
    
    if not np.any(near_hover):
        print("Warning: No near-hover samples found; cannot reliably estimate mass.")
        return None

    # For near-hover, we expect vz_dot to be very small so that:
    # m ~ Fz / (g * cos(phi) * cos(theta))
    m_samples = Fz[near_hover] / (g * np.cos(phi[near_hover]) * np.cos(theta[near_hover]))
    m_est = np.median(m_samples)
    
    print("Median mass from near-hover samples:", m_est)
    return m_est
def residual_inertias(params, states, controls_raw):
    """
    Residual function for inertias.
    
    For each time step, the model for roll dynamics is:
      p_dot = (1/I_xx) * (tau_x + (I_yy-I_zz)*q*r)
    Similarly for pitch and yaw:
      q_dot = (1/I_yy) * (tau_y + (I_zz-I_xx)*p*r)
      r_dot = (1/I_zz) * (tau_z + (I_xx-I_yy)*p*q)
    
    Rearranged, we define residuals:
      r1 = tau_x - I_xx * p_dot + (I_yy - I_zz)*q*r
      r2 = tau_y - I_yy * q_dot + (I_zz - I_xx)*p*r
      r3 = tau_z - I_zz * r_dot + (I_xx - I_yy)*p*q
    
    This function returns a concatenated residual vector.
    """
    I_xx, I_yy, I_zz = params
    # Angular velocities
    p = states[:, 9]
    q = states[:, 10]
    r = states[:, 11]
    # Compute derivatives of angular velocities:
    p_dot = np.gradient(p, states[:,0])  # We do not have explicit time here so use spacing from first state.
    # Instead, better: assume uniform spacing and use np.gradient with respect to time vector.
    # (We will pass time separately if needed; here we assume similar sampling rate.)
    # For clarity, we assume dt is constant. Here we re-use np.gradient on p, q, r
    dt = np.mean(np.diff(np.linspace(0, len(p), len(p))))
    p_dot = np.gradient(p, dt)
    q_dot = np.gradient(q, dt)
    r_dot = np.gradient(r, dt)
    # Torques from controls:
    tau_x = controls_raw[:, 3]
    tau_y = controls_raw[:, 4]
    tau_z = controls_raw[:, 5]
    r1 = tau_x - I_xx * p_dot + (I_yy - I_zz) * q * r
    r2 = tau_y - I_yy * q_dot + (I_zz - I_xx) * p * r
    r3 = tau_z - I_zz * r_dot + (I_xx - I_yy) * p * q
    return np.concatenate((r1, r2, r3))

def estimate_inertias(states, controls_raw, I_guess):
    """
    Estimate the three inertias by minimizing the residuals from the angular dynamics.
    """
    # We use least_squares to solve the nonlinear regression.
    result = least_squares(residual_inertias, I_guess, args=(states, controls_raw))
    return result.x

def estimate_normalization_factors(t, states, controls_norm, controls_raw, m_est, I_est, g=9.81):
    """
    Estimate the scaling factors (for Fx, Fy, Fz, tau_x, tau_y, tau_z) used
    in your normalization function.
    
    We assume that during near-hover conditions (small angles and low velocities)
    the following approximations hold:
    
      Fx_raw = m_est * ax,
      Fy_raw = m_est * ay,
      Fz_raw = m_est * (az + g),
      tau_x_raw = I_est[0] * p_dot,
      tau_y_raw = I_est[1] * q_dot,
      tau_z_raw = I_est[2] * r_dot.
    
    Since the bagfile recorded normalized controls (controls_norm) and you have
    already computed raw controls (controls_raw) using your nominal factors, you can
    form for each channel:
    
       scaling = normalized / raw
    
    and then average over “good” indices.
    """
    # Identify near-hover segments: small roll and pitch and small velocities.
    phi = states[:, 6]
    theta = states[:, 7]
    vx = states[:, 3]
    vy = states[:, 4]
    vz = states[:, 5]
    near_hover = (np.abs(phi) < 0.1) & (np.abs(theta) < 0.1) & (np.abs(vx) < 0.1) & (np.abs(vy) < 0.1)
    if not np.any(near_hover):
        print("Warning: no near-hover segments found for normalization estimation.")
        near_hover = np.ones_like(vx, dtype=bool)
    
    # Compute accelerations
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    vz_dot = np.gradient(vz, t)
    # For torques, compute angular accelerations (p_dot, q_dot, r_dot)
    p = states[:, 9]
    q = states[:, 10]
    r = states[:, 11]
    dt = np.mean(np.diff(t))
    p_dot = np.gradient(p, dt)
    q_dot = np.gradient(q, dt)
    r_dot = np.gradient(r, dt)
    
    # Raw controls from dynamics (expected raw command) computed from measured accelerations:
    # For Fx: F_raw = m_est * ax, similarly Fy.
    Fx_est = m_est * ax
    Fy_est = m_est * ay
    # For Fz: from vertical dynamics at hover, az ~ (Fz/m) - g, so Fz = m*(vz_dot + g)
    Fz_est = m_est * (vz_dot + g)
    # For torques:
    tau_x_est = I_est[0] * p_dot
    tau_y_est = I_est[1] * q_dot
    tau_z_est = I_est[2] * r_dot
    
    # Compute per-sample ratios (only over near-hover indices and avoiding near-zero denominators)
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
        description="Estimate mass, inertias, and 6D normalization factors using your acados MPC dynamics model from flight data.")
    parser.add_argument("--bagfiles", nargs="+", required=True,
                        help="List of bagfile specs in the format: path,begin,end (e.g., data1.bag,0,100 data2.bag,120,200)")
    parser.add_argument("--state_topic", default="/odometry", help="State topic (default: /odometry)")
    parser.add_argument("--control_topic", default="/mppi_debug/control_cmd", help="Control topic (default: /mppi_debug/control_cmd)")
    args = parser.parse_args()

    all_t = []
    all_states = []
    all_controls_norm = []  # these are the normalized controls as recorded

    for spec in args.bagfiles:
        try:
            bag_path, t_begin_str, t_end_str = spec.split(",")
            t_begin = float(t_begin_str)
            t_end = float(t_end_str)
        except Exception as e:
            print("Error parsing bagfile spec '{}': {}".format(spec, e))
            continue
        print("Processing bagfile: {} (relative t = {} to {})".format(bag_path, t_begin, t_end))
        t_state, states, t_control, controls = load_data_from_bagfile(bag_path, args.state_topic, args.control_topic, t_begin, t_end)
        if len(t_state) < 2 or len(t_control) < 2:
            print("Not enough data in bagfile:", bag_path)
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
        all_t.append(t_state)
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

    # Create an instance of your MPC to access nominal parameters.
    params = {
        'inertia': [0.115125971, 0.116524229, 0.230387752],
        'mass': 7.00,  # nominal mass used for normalization
        'gravity': 9.81,
        'max_force': 20.0,
        'max_torque': 0.05,
        'control_weight': 0.005,
        'tracking_weight_pos': 10,
        'tracking_weight_vel': 3,
        'tracking_weight_att': 80,
        'tracking_weight_ang_vel': 50,
        'smoothness_weight': 0.01,
        'dt': 0.3
    }
    mpc = OneStepMPC(params)

    # De-normalize the controls using the nominal mass (7.00 kg)
    controls_all_raw = denormalize_controls(controls_all_norm, nominal_mass=params["mass"])

    # Estimate mass using vertical dynamics.
    m_est = estimate_mass(t_all, states_all, controls_all_raw, g=params["gravity"])
    print("Estimated mass: {:.3f} kg".format(m_est))

    # Estimate inertias using angular dynamics.
    I_guess = np.array(params["inertia"])  # initial guess from nominal parameters
    I_est = estimate_inertias(states_all, controls_all_raw, I_guess)
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
