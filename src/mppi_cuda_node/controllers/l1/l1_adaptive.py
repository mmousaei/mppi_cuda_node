import numpy as np
import casadi as cs
from scipy.signal import savgol_filter  # Requires SciPy

class L1AdaptiveController:
    """
    A simple L1 adaptive controller that estimates (matched) uncertainties by
    comparing the measured state derivative (via finite differences) with the
    predicted derivative from the nominal model.
    The computed compensation (after applying a Savitzky–Golay filter, a low-pass filter, and scaling)
    is added to the baseline control from the MPC.
    
    This version uses separate adaptation gains for the position and attitude channels.
    """
    def __init__(self, params, f_nominal):
        # Separate adaptation gains for position and attitude.
        self.adaptation_gain_pos_vertical = params.get("l1_adaptation_gain_pos_vertical", 10.0)
        self.adaptation_gain_pos_horizontal = params.get("l1_adaptation_gain_pos_horizontal", 10.0)
        self.adaptation_gain_att = params.get("l1_adaptation_gain_att", 10.0)
        # Low-pass filter cutoff frequency (Hz) for additional smoothing.
        self.filter_cutoff = params.get("l1_filter_cutoff", 5.0)
        # Nominal dynamics function (CasADi Function taking (x,u) and returning f(x,u)).
        self.f_nominal = f_nominal
        # Initialize the (low-pass filtered) adaptive control signal (6D vector).
        self.adapted_control = np.zeros(6)
        # Define an approximate input matrix B for the “matched uncertainty” channels.
        mass = params["mass"]
        I_xx, I_yy, I_zz = params["inertia"]
        self.B = np.zeros((12, 6))
        # Control appears in the acceleration (indices 3,4,5) and angular acceleration (indices 9,10,11)
        self.B[3, 0] = 1.0 / mass
        self.B[4, 1] = 1.0 / mass
        self.B[5, 2] = 1.0 / mass
        self.B[9, 3] = 1.0 / I_xx
        self.B[10, 4] = 1.0 / I_yy
        self.B[11, 5] = 1.0 / I_zz
        # Pre-compute the pseudoinverse of B.
        self.B_pinv = np.linalg.pinv(self.B)
        
        # Parameters for the Savitzky–Golay filter.
        # The window length must be an odd number.
        self.savgol_window = params.get("savgol_window", 5)
        self.savgol_polyorder = params.get("savgol_polyorder", 2)
        # Buffer to store recent disturbance estimates.
        self.disturbance_buffer = []

    def update(self, state, prev_state, u_mpc, dt):
        """
        Update the adaptive control signal.
        Inputs:
          state      : current state (12D)
          prev_state : previous state (12D)
          u_mpc      : baseline control from the MPC (6D)
          dt         : simulation integration timestep
        Returns:
          u_adapt    : adaptive control signal (6D) to be added to u_mpc.
        """
        # Estimate the state derivative using finite differences.
        x_dot_meas = (state - prev_state) / dt
        # Predict the state derivative using the nominal model (evaluated at prev_state and u_mpc).
        f_nom = np.array(self.f_nominal(prev_state, u_mpc)).flatten()
        # Estimate the disturbance as the difference between measured and predicted derivatives.
        disturbance_est = x_dot_meas - f_nom
        # Project the disturbance estimate into the control channel via the pseudoinverse of B.
        u_dist_est = self.B_pinv @ disturbance_est
        
        # Append the current disturbance estimate to the buffer.
        self.disturbance_buffer.append(u_dist_est)
        # Keep the buffer length fixed to the savgol window length.
        if len(self.disturbance_buffer) > self.savgol_window:
            self.disturbance_buffer.pop(0)
        
        # # Apply the Savitzky–Golay filter if the buffer is full.
        # if len(self.disturbance_buffer) == self.savgol_window:
        #     buffer_array = np.array(self.disturbance_buffer)  # shape: (window_length, 6)
        #     filtered_buffer = savgol_filter(buffer_array, window_length=self.savgol_window,
        #                                     polyorder=self.savgol_polyorder, axis=0)
        #     u_dist_est_filtered = filtered_buffer[-1, :]
        # else:
        #     u_dist_est_filtered = u_dist_est
        
        # Apply a first-order low-pass filter.
        tau = 1.0 / (2 * np.pi * self.filter_cutoff)
        alpha = dt / (dt + tau)
        self.adapted_control = (1 - alpha) * self.adapted_control + alpha * u_dist_est
        
        # Apply separate adaptation gains:
        # First 3 elements (position) and last 3 elements (attitude).
        u_adapt = np.zeros(6)
        u_adapt[0:2] = self.adaptation_gain_pos_horizontal * self.adapted_control[0:2]
        u_adapt[2] = self.adaptation_gain_pos_vertical * self.adapted_control[2]
        u_adapt[3:6] = self.adaptation_gain_att * self.adapted_control[3:6]
        
        return u_adapt
    



