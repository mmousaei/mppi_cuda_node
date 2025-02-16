import numpy as np
import casadi as cs
from scipy.signal import savgol_filter  # Requires SciPy

class L1AdaptiveController:
    """
    An L1 adaptive controller that estimates (matched) uncertainties using an EKF observer
    for the augmented state z = [x; d], where d is the disturbance (model mismatch) in the state derivative.
    The computed compensation (after applying a Savitzky–Golay filter, a low-pass filter, and channel‐specific gains)
    is added to the baseline control from the MPC.
    """
    def __init__(self, params, f_nominal):
        # --- Adaptive Gains for translation (separated into horizontal and vertical) and attitude.
        self.adaptation_gain_pos_vertical = params.get("l1_adaptation_gain_pos_vertical", 10.0)
        self.adaptation_gain_pos_horizontal = params.get("l1_adaptation_gain_pos_horizontal", 10.0)
        self.adaptation_gain_att = params.get("l1_adaptation_gain_att", 10.0)
        # Use separate filter cutoff frequencies in Hz for translational and rotational channels.
        self.filter_cutoff_trans = params.get("l1_filter_cutoff_trans", 5.0)
        self.filter_cutoff_rot   = params.get("l1_filter_cutoff_rot", 5.0)
        self.tau_trans = 1.0 / (2 * np.pi * self.filter_cutoff_trans)
        self.tau_rot   = 1.0 / (2 * np.pi * self.filter_cutoff_rot)
        
        # Nominal dynamics function (CasADi Function taking (x,u) and returning f(x,u)).
        self.f_nominal = f_nominal
        
        # --- Build a CasADi function to compute the Jacobian of f_nominal with respect to x.
        x_sym = cs.MX.sym('x', 12)
        u_sym = cs.MX.sym('u', 6)
        f_expr = self.f_nominal(x_sym, u_sym)
        self.A_func = cs.Function('A_func', [x_sym, u_sym], [cs.jacobian(f_expr, x_sym)])
        
        # Initialize the (filtered) adaptive control signal (6D vector).
        self.adapted_control = np.zeros(6)
        
        # --- Define the nominal input matrix B for the matched uncertainty channels.
        mass = params["mass"]
        I_xx, I_yy, I_zz = params["inertia"]
        self.B = np.zeros((12, 6))
        # Translational channels: acceleration indices 3,4,5.
        self.B[3, 0] = 1.0 / mass
        self.B[4, 1] = 1.0 / mass
        self.B[5, 2] = 1.0 / mass
        # Attitude channels: angular acceleration indices 9,10,11.
        self.B[9, 3] = 1.0 / I_xx
        self.B[10, 4] = 1.0 / I_yy
        self.B[11, 5] = 1.0 / I_zz
        self.B_pinv = np.linalg.pinv(self.B)
        
        # --- Parameters for the Savitzky–Golay filter.
        self.savgol_window = params.get("savgol_window", 5)  # Must be odd.
        self.savgol_polyorder = params.get("savgol_polyorder", 2)
        self.disturbance_buffer = []

        # --- EKF Initialization for the augmented state z = [x; d].
        # We assume state dimension n = 12, disturbance dimension n = 12, so total dim = 24.
        self.n = 12
        self.z_dim = 2 * self.n
        self.ekf_state = np.zeros(self.z_dim)  # Initially, x_est = 0 and d_est = 0.
        self.P = np.eye(self.z_dim) * 1e-3    # Initial covariance.
        # Process noise covariance Q (can be tuned).
        self.Q = params.get("ekf_Q", np.eye(self.z_dim) * 1e-2)
        # Measurement noise covariance R (for measurements of x).
        self.R = params.get("ekf_R", np.eye(self.n) * 1e-5)

    def _ekf_update(self, measurement, u, dt):
        """
        EKF prediction and update for the augmented state z = [x; d].
        Process model:
          x_{k+1} = x_k + dt*( f_nominal(x_k, u) + d_k )
          d_{k+1} = d_k
        Measurement model:
          y = x
        Updates self.ekf_state and self.P.
        Returns:
          d_hat: The estimated disturbance (model mismatch) in state derivative (12D).
        """
        n = self.n
        z = self.ekf_state  # z = [x; d]
        x = z[:n]
        d = z[n:]
        
        # Predict step:
        # Compute nominal derivative using the current estimate x.
        f_val = np.array(self.f_nominal(x, u)).flatten()  # f_nominal(x,u)
        x_pred = x + dt * (f_val + d)
        d_pred = d  # d is assumed constant.
        z_pred = np.concatenate([x_pred, d_pred])
        
        # Compute the Jacobian F = dF/dz.
        # F = [I + dt*A, dt*I; 0, I]
        A = np.array(self.A_func(x, u))
        I = np.eye(n)
        F = np.block([
            [I + dt * A, dt * I],
            [np.zeros((n, n)), I]
        ])
        
        # Covariance prediction.
        P_pred = F @ self.P @ F.T + self.Q
        
        # Measurement model: y = x, so H = [I, 0]
        H = np.hstack([np.eye(n), np.zeros((n, n))])
        
        # Kalman gain.
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Innovation.
        y_pred = z_pred[:n]
        innovation = measurement - y_pred
        
        # Update step.
        z_upd = z_pred + K @ innovation
        P_upd = (np.eye(self.z_dim) - K @ H) @ P_pred
        
        # Save updated state.
        self.ekf_state = z_upd
        self.P = P_upd
        
        # Return the disturbance estimate d_hat.
        d_hat = z_upd[n:]
        return d_hat

    def update(self, state, u_mpc, dt):
        """
        Update the adaptive control signal using the EKF estimate of the disturbance.
        Inputs:
          state   : measured current state (12D)
          u_mpc   : baseline MPC control (6D)
          dt      : simulation timestep
        Returns:
          u_adapt : adaptive control signal (6D) to be added to u_mpc.
        """
        # Use the EKF to obtain a robust estimate of the disturbance d (i.e. model mismatch in x_dot).
        d_hat = self._ekf_update(measurement=state, u=u_mpc, dt=dt)
        
        # Project the disturbance (which is in state derivative space) into the control space.
        # (This is analogous to: u_dist_est = B_pinv @ (x_dot_meas - f_nominal))
        u_dist_est = self.B_pinv @ d_hat
        
        # Append the current u_dist_est to the Savitzky–Golay filter buffer.
        self.disturbance_buffer.append(u_dist_est)
        if len(self.disturbance_buffer) > self.savgol_window:
            self.disturbance_buffer.pop(0)
        if len(self.disturbance_buffer) == self.savgol_window:
            buffer_array = np.array(self.disturbance_buffer)  # shape: (window_length, 6)
            filtered_buffer = savgol_filter(buffer_array, window_length=self.savgol_window,
                                            polyorder=self.savgol_polyorder, axis=0)
            u_dist_est_filtered = filtered_buffer[-1, :]
        else:
            u_dist_est_filtered = u_dist_est

        # Apply a first-order low-pass filter to smooth the adaptive control signal.
        # Translational channels (indices 0,1,2):
        alpha_trans = dt / (dt + self.tau_trans)
        self.adapted_control[0:3] = ((1 - alpha_trans) * self.adapted_control[0:3] + alpha_trans * u_dist_est_filtered[0:3])
        # Rotational channels (indices 3,4,5):
        alpha_rot = dt / (dt + self.tau_rot)
        self.adapted_control[3:6] = ((1 - alpha_rot) * self.adapted_control[3:6] + alpha_rot * u_dist_est_filtered[3:6])

        # Apply separate adaptation gains:
        # For translation: indices 0-1 are horizontal, index 2 is vertical.
        u_adapt = np.zeros(6)
        u_adapt[0:2] = self.adaptation_gain_pos_horizontal * self.adapted_control[0:2]
        u_adapt[2]   = self.adaptation_gain_pos_vertical * self.adapted_control[2]
        # For attitude: indices 3-5.
        u_adapt[3:6] = self.adaptation_gain_att * self.adapted_control[3:6]

        return u_adapt

    def reset(self, state=None):
        """
        Reset the internal state of the adaptive controller.
        This resets:
          - The EKF augmented state (z = [x; d]): if a measured state (12D) is provided, x is set to that and d to zero;
            otherwise, the entire state is reset to zero.
          - The covariance matrix P is reinitialized.
          - The disturbance buffer is cleared.
          - The filtered adaptive control signal is reset.
        Args:
            state (np.array, optional): Measured current state (12D) to initialize the observer.
        """
        self.adapted_control = np.zeros(6)
        self.disturbance_buffer = []
        if state is not None:
            self.ekf_state = np.concatenate([state, np.zeros(self.n)])
        else:
            self.ekf_state = np.zeros(self.z_dim)
        self.P = np.eye(self.z_dim) * 1e-3