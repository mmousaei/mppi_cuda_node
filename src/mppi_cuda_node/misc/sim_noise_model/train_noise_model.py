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

# Import your MPC class (ensure PYTHONPATH is set appropriately)
from mppi_cuda_node.controllers.mpc.acados.acados_mpc import MPC

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

class NoiseNet(nn.Module):
    """
    A simple feedforward neural network to learn the residual (model mismatch)
    given an input that concatenates state (12D) and control (6D) for an 18D input.
    """
    def __init__(self, input_dim, output_dim):
        super(NoiseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser(
        description="Learn a Noise Model (MPC Model Mismatch) using a Neural Network from Bagfile Data"
    )
    parser.add_argument("bag_dir", help="Directory containing bagfiles")
    parser.add_argument("--state_topic", default="/odometry", help="Topic for state messages")
    parser.add_argument("--control_topic", default="/mppi_debug/control_cmd", help="Topic for control messages")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lambda_smooth", type=float, default=1e-3, help="Temporal smoothness regularizer weight")
    # Parameters for smoothing the state before computing residuals.
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

    # Get list of bagfiles in the specified directory.
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
    states = np.concatenate(all_states, axis=0)   # shape (N, 12)
    controls = np.concatenate(all_controls, axis=0) # shape (M, 6)
    t_state = np.concatenate(all_t_state, axis=0)   # shape (N,)
    t_control = np.concatenate(all_t_control, axis=0) # shape (M,)

    print("Total state samples loaded:", len(states))
    print("Total control samples loaded:", len(controls))

    # Smooth the raw states to reduce spikiness.
    smooth_states = savgol_filter(states, window_length=args.smooth_window, polyorder=args.smooth_poly, axis=0)

    # Interpolate control data to align with state timestamps.
    controls_aligned = np.zeros((len(t_state), controls.shape[1]))
    if len(t_control) > 1 and len(t_state) > 1:
        for ch in range(controls.shape[1]):
            interp_func = interp1d(t_control, controls[:, ch], kind='linear', fill_value="extrapolate")
            controls_aligned[:, ch] = interp_func(t_state)
    else:
        controls_aligned = controls

    # Prepare dataset for neural network training.
    # For each consecutive smoothed state sample, compute:
    #   predicted_state = smooth_state[i] + dt * f(x[i], u[i])
    #   residual = smooth_state[i+1] - predicted_state
    # NN input: [smooth_state[i]; control_aligned[i]] (18D), target: residual (12D)
    mpc = MPC(mpc_params)
    f_expl = mpc.model.f_expl_expr
    f_func = cs.Function("f_func", [mpc.model.x, mpc.model.u], [f_expl])

    X_list = []
    y_list = []
    N = len(smooth_states)
    for i in range(N - 1):
        dt = t_state[i+1] - t_state[i]
        if dt <= 0:
            continue
        x_i = smooth_states[i]           # 12D vector (smoothed)
        u_i = controls_aligned[i]        # 6D vector, aligned with state timestamps
        f_val = np.array(f_func(x_i, u_i)).flatten()  # 12D derivative from MPC model
        predicted_state = x_i + dt * f_val
        residual = smooth_states[i+1] - predicted_state  # 12D residual
        input_vec = np.concatenate([x_i, u_i], axis=0)     # 18D vector
        X_list.append(input_vec)
        y_list.append(residual)

    if len(X_list) == 0:
        print("No valid consecutive samples to compute residuals.")
        return

    X = np.array(X_list)  # shape (M, 18)
    y = np.array(y_list)  # shape (M, 12)
    print("Total training samples:", X.shape[0])

    # Convert data to PyTorch tensors.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Split into training and testing sets, but do NOT shuffle to preserve temporal order.
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, shuffle=False)

    # Create and train the neural network.
    model = NoiseNet(input_dim=18, output_dim=12)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lambda_smooth = args.lambda_smooth

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        # Primary loss: mean squared error.
        mse_loss = criterion(outputs, y_train)
        # Temporal smoothness regularizer: penalize differences between consecutive predictions.
        # Assuming X_train is temporally ordered.
        if outputs.size(0) > 1:
            smooth_loss = torch.mean((outputs[1:] - outputs[:-1])**2) * 1000
        else:
            smooth_loss = 0.0
        total_loss = mse_loss + lambda_smooth * smooth_loss
        total_loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
            print("Epoch [{}/{}], Train Loss: {:.6f}, Test Loss: {:.6f}, Smooth Loss: {:.6f}".format(
                epoch+1, num_epochs, mse_loss.item(), test_loss.item(), smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss))

    # Save the trained model.
    model_file = "noise_model_nn.pt"
    torch.save(model.state_dict(), model_file)
    print("Neural network noise model saved to", model_file)

    # Plot an example: compare actual vs. predicted residual for channel 0.
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
    plt.figure()
    plt.plot(y[:, 0], label="Actual Residual (channel 0)")
    plt.plot(y_pred[:, 0], label="Predicted Residual (channel 0)", alpha=0.7)
    plt.xlabel("Sample")
    plt.ylabel("Residual Value")
    plt.title("Neural Network Noise Model (MPC Mismatch) - Channel 0")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
