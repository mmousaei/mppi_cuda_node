import numpy as np
import casadi as cs
from nav_msgs.msg import Odometry
import rospy

class L1AdaptiveController:
    """
    A revised L1 adaptive controller using a high-gain observer for disturbance (model mismatch) estimation.
    
    The controller maintains a state predictor:
      hat_x_dot = f_nominal(hat_x, u_mpc) + d_hat + L*(x - hat_x)
    and an adaptation law:
      sigma_dot = gamma * (x - hat_x)
    The disturbance estimate d_hat is obtained by low-pass filtering sigma.
    This d_hat is then projected to the control space via the nominal input matrix B.
    
    Tuning parameters:
      - l1_gamma: adaptation gain (how fast the estimator reacts to the prediction error)
      - l1_observer_gain: observer gain L (typically chosen high)
      - l1_filter_cutoff: cutoff frequency for the low-pass filter (determines the filter time constant tau)
      - Channel-specific gains: l1_adaptation_gain_pos_horizontal, l1_adaptation_gain_pos_vertical, l1_adaptation_gain_att
    """
    def __init__(self, params, f_nominal):
        # Adaptation gains for each channel
        self.adaptation_gain_pos_vertical   = params.get("l1_adaptation_gain_pos_vertical", 10.0)
        self.adaptation_gain_pos_horizontal = params.get("l1_adaptation_gain_pos_horizontal", 10.0)
        self.adaptation_gain_att            = params.get("l1_adaptation_gain_att", 10.0)
        
        # Low-pass filter: cutoff frequency in Hz, time constant tau = 1/(2*pi*fc)
        filter_cutoff = params.get("l1_filter_cutoff", 10.0)
        self.tau = 1.0 / (2 * np.pi * filter_cutoff)
        
        # Adaptation law gain (how quickly to integrate the prediction error)
        self.gamma = params.get("l1_gamma", 10.0)
        
        # Observer gain L (a high gain matrix to force hat_x to follow x)
        # For simplicity, we use L = alpha * I, with alpha tunable.
        alpha = params.get("l1_observer_gain", 10.0)
        self.L = np.eye(12) * alpha
        
        # Nominal dynamics function (CasADi function taking (x, u) and returning f_nominal(x, u))
        self.f_nominal = f_nominal
        
        # Dimensions
        self.n = 12
        
        # Initialize the state predictor, the adaptation law variable (sigma), and the filtered disturbance (d_hat)
        self.hat_x = np.zeros(self.n)   # state predictor (initialized later via reset)
        self.sigma  = np.zeros(self.n)   # fast adaptation signal (unfiltered)
        self.d_hat  = np.zeros(self.n)   # low-pass filtered disturbance estimate
        
        # --- Define the nominal input matrix B for the matched uncertainty channels ---
        mass = params["mass"]
        I_xx, I_yy, I_zz = params["inertia"]
        self.B = np.zeros((12, 6))
        # Translational channels: acceleration indices 3,4,5.
        self.B[3, 0] = 1.0 / mass
        self.B[4, 1] = 1.0 / mass
        self.B[5, 2] = 1.0 / mass
        # Attitude channels: angular acceleration indices 9,10,11.
        self.B[9, 3]  = 1.0 / I_xx
        self.B[10, 4] = 1.0 / I_yy
        self.B[11, 5] = 1.0 / I_zz
        self.B_pinv = np.linalg.pinv(self.B)
        
        # For monitoring the final adaptive control signal (6D)
        self.adapted_control = np.zeros(6)
        
        # Optional: you may add logging or debug publishers if desired.
        self.debug_publishers = {}

    def update(self, state, u_mpc, dt):
        """
        Update the adaptive control signal.
        Inputs:
          state   : measured current state (12D)
          u_mpc   : baseline MPC control (6D)
          dt      : timestep
        Returns:
          u_adapt : adaptive control signal (6D) to be added to u_mpc.
        """
        # On the first iteration, initialize the state predictor with the measured state.
        if np.linalg.norm(self.hat_x) < 1e-6:
            self.hat_x = state.copy()
        
        # -----------------------------
        # 1. State Predictor Update
        # -----------------------------
        # Compute nominal dynamics based on the predicted state.
        f_val = np.array(self.f_nominal(self.hat_x, u_mpc)).flatten()  # 12D vector
        prediction_error = state - self.hat_x  # error between measured state and predicted state
        
        # Update the state predictor using Euler integration:
        # hat_x_dot = f_nominal(hat_x,u_mpc) + d_hat + L*(state - hat_x)
        self.hat_x = self.hat_x + dt * (f_val + self.d_hat + self.L.dot(prediction_error))
        
        # -----------------------------
        # 2. Adaptation Law Update
        # -----------------------------
        # Update the fast adaptation signal sigma using the prediction error:
        # sigma_dot = gamma * (state - hat_x)
        self.sigma = self.sigma + dt * self.gamma * (state - self.hat_x)
        
        # -----------------------------
        # 3. Low-Pass Filter the Adaptation Signal
        # -----------------------------
        # Use a simple discrete-time low-pass filter to obtain d_hat from sigma.
        # d_hat = alpha* sigma + (1 - alpha)* previous_d_hat, with alpha = dt/(tau + dt)
        alpha = dt / (self.tau + dt)
        self.d_hat = alpha * self.sigma + (1 - alpha) * self.d_hat
        
        # -----------------------------
        # 4. Project Disturbance Estimate into Control Space
        # -----------------------------
        # The nominal input matrix B maps control to state-acceleration.
        # We use its pseudoinverse to project the disturbance estimate (in state derivative space)
        # into a control correction.
        u_dist_est = self.B_pinv.dot(self.d_hat)
        
        # -----------------------------
        # 5. Apply Channel-Specific Adaptation Gains and Clipping
        # -----------------------------
        u_adapt = np.zeros(6)
        # For translation: indices 0-1 are horizontal, index 2 is vertical.
        u_adapt[0:2] = self.adaptation_gain_pos_horizontal * u_dist_est[0:2]
        u_adapt[2]   = self.adaptation_gain_pos_vertical   * u_dist_est[2]
        # For attitude: indices 3-5.
        u_adapt[3:6] = self.adaptation_gain_att            * u_dist_est[3:6]
        
        # Optionally, clip the adaptive control signal to prevent excessive correction.
        fmax = 10
        mmax = 0.1
        umax = np.array([fmax, fmax, 8*fmax, mmax, mmax, mmax])
        umin = np.array([-fmax, -fmax, 0, -mmax, -mmax, -mmax])
        u_adapt = np.clip(u_adapt, 0.3 * umin, 0.3 * umax)
        
        self.adapted_control = u_adapt
        return u_adapt

    def reset(self, state=None):
        """
        Reset the internal state of the adaptive controller.
        If a measured state is provided, initialize the predictor with it.
        """
        if state is not None:
            self.hat_x = state.copy()
        else:
            self.hat_x = np.zeros(self.n)
        self.sigma  = np.zeros(self.n)
        self.d_hat  = np.zeros(self.n)
        self.adapted_control = np.zeros(6)
