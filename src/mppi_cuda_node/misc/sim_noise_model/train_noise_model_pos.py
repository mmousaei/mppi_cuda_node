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

# Define an LSTM-based model for sequence prediction.
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

def main():
    parser = argparse.ArgumentParser(
        description="Learn a Noise Model (MPC Model Mismatch for x,y,z) using an LSTM from Bagfile Data"
    )
    parser.add_argument("bag_dir", help="Directory containing bagfiles")
    parser.add_argument("--state_topic", default="/odometry", help="Topic for state messages")
    parser.add_argument("--control_topic", default="/mppi_debug/control_cmd", help="Topic for control messages")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length for LSTM")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
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

    # Prepare dataset for sequence learning.
    # We will only consider x, y, z positions. For each time step,
    # our input is [position (3D); control force (3D)] and target is the residual in position.
    # The predicted state is computed using only the translational part of the MPC model.
    mpc = MPC(mpc_params)
    f_expl = mpc.model.f_expl_expr
    f_func = cs.Function("f_func", [mpc.model.x, mpc.model.u], [f_expl])

    inputs = []
    targets = []
    N_total = len(smooth_states)
    for i in range(N_total - 1):
        dt = t_state[i+1] - t_state[i]
        if dt <= 0:
            continue
        # Use only the position part (first 3 elements) from the smoothed state.
        x_i = smooth_states[i][:3]           # 3D vector (position)
        # And use only the force part (first 3 elements) from the aligned control.
        u_i = controls_aligned[i][:3]          # 3D vector (force)
        # Evaluate model derivative (we only take the translational part).
        f_val_full = np.array(f_func(smooth_states[i], controls_aligned[i])).flatten()
        f_val = f_val_full[:3]  # Only the first 3 elements for x, y, z acceleration.
        predicted_pos = x_i + dt * f_val
        residual = smooth_states[i+1][:3] - predicted_pos  # 3D residual for position
        # Input: concatenation of position and force (6D)
        input_vec = np.concatenate([x_i, u_i], axis=0)
        inputs.append(input_vec)
        targets.append(residual)

    inputs = np.array(inputs)   # shape (M, 6)
    targets = np.array(targets) # shape (M, 3)
    print("Total individual training samples:", inputs.shape[0])

    # Now form sequences of length seq_len and predict the residual at the next time step.
    seq_len = args.seq_len
    X_seq = []
    y_seq = []
    for i in range(len(inputs) - seq_len):
        X_seq.append(inputs[i:i+seq_len])
        y_seq.append(targets[i+seq_len])
    X_seq = np.array(X_seq)  # shape (M_seq, seq_len, 6)
    y_seq = np.array(y_seq)  # shape (M_seq, 3)
    print("Total sequence training samples:", X_seq.shape[0])

    # Convert data to PyTorch tensors.
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)

    # Split into training and testing sets (preserving order).
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, shuffle=False)

    # Create the LSTM model.
    # Input dimension is 6 (3 for position, 3 for force), output dimension is 3.
    model = ResidualLSTM(input_dim=6, hidden_dim=args.hidden_dim, num_layers=args.num_layers, output_dim=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)  # outputs shape: (batch, 3)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
            print("Epoch [{}/{}], Train Loss: {:.6f}, Test Loss: {:.6f}".format(
                epoch+1, num_epochs, loss.item(), test_loss.item()))

    # Save the trained model.
    model_file = "noise_model_nn.pt"
    torch.save(model.state_dict(), model_file)
    print("Neural network noise model saved to", model_file)

    # Plot an example: compare actual vs. predicted residual for x (channel 0).
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
    plt.figure()
    plt.plot(y_seq[:, 0], label="Actual Residual (x)")
    plt.plot(y_pred[:, 0], label="Predicted Residual (x)", alpha=0.7)
    plt.xlabel("Sequence Sample")
    plt.ylabel("Residual Value")
    plt.title("LSTM Noise Model (MPC Mismatch) for x")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
