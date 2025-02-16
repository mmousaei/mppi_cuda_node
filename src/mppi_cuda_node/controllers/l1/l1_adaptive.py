import numpy as np
import casadi as cs

class L1AdaptiveController:
    """
    An improved L1 adaptive controller implementation with separate filtering for 
    translational and rotational dynamics.
    
    Assumptions:
      - The plant dynamics are:
            x_dot = f_nominal(x, u) + B * sigma,
        where sigma is an unknown but matched uncertainty.
      - A state predictor is implemented:
            x_hat_dot = f_nominal(x_hat, u_mpc) + B * sigma_hat.
      - The adaptation law uses the prediction error, mapped via B_pinv.
      - First-order low-pass filters (with separate cutoff frequencies) are applied 
        to the translational and rotational adaptive control signals.
      - Projection (saturation) is used to bound the uncertainty estimate.
      
    The final adaptive control is added to the baseline (e.g. MPC) control.
    """
    def __init__(self, params, f_nominal):
        # --- Adaptation law gains (for estimating the matched uncertainty)
        # Separate gains for horizontal position, vertical position, and attitude.
        self.gain_pos_horizontal = params.get("l1_adaptation_gain_pos_horizontal", 10.0)
        self.gain_pos_vertical   = params.get("l1_adaptation_gain_pos_vertical", 10.0)
        self.gain_att            = params.get("l1_adaptation_gain_att", 10.0)
        
        # --- Low-pass filter settings
        # Use separate filter cutoff frequencies in Hz for translational and rotational channels.
        self.filter_cutoff_trans = params.get("l1_filter_cutoff_trans", 5.0)
        self.filter_cutoff_rot   = params.get("l1_filter_cutoff_rot", 5.0)
        # Convert cutoff frequencies to time constants: tau = 1/(2*pi*cutoff)
        self.tau_trans = 1.0 / (2 * np.pi * self.filter_cutoff_trans)
        self.tau_rot   = 1.0 / (2 * np.pi * self.filter_cutoff_rot)
        
        # --- Nominal dynamics function (CasADi function f_nominal(x, u) -> x_dot)
        self.f_nominal = f_nominal
        
        # --- Construct the input matrix B for the matched uncertainty channels.
        mass = params["mass"]
        I_xx, I_yy, I_zz = params["inertia"]
        self.B = np.zeros((12, 6))
        # Force channels (accelerations): indices 3, 4, 5.
        self.B[3, 0] = 1.0 / mass
        self.B[4, 1] = 1.0 / mass
        self.B[5, 2] = 1.0 / mass
        # Moment channels (angular accelerations): indices 9, 10, 11.
        self.B[9, 3] = 1.0 / I_xx
        self.B[10, 4] = 1.0 / I_yy
        self.B[11, 5] = 1.0 / I_zz
        
        # Precompute the pseudoinverse of B (used for mapping state error to the uncertainty channels).
        self.B_pinv = np.linalg.pinv(self.B)
        
        # --- Projection bounds for the adaptation estimate.
        # This ensures that the estimated uncertainty remains bounded.
        fmax = 10
        mmax = 0.2
        sigma_max_f = np.array([fmax, fmax, fmax])
        sigma_max_m = np.array([mmax, mmax, mmax])
        self.sigma_max = np.concatenate([sigma_max_f, sigma_max_m])
        
        # --- State predictor initialization.
        # (x_hat will be initialized to the measured state at the first update.)
        self.x_hat = None
        
        # --- Initialize the estimated matched uncertainty (6D vector).
        self.adaptation_estimate = np.zeros(6)
        
        # --- Initialize the filtered adaptive control signal (6D vector).
        self.u_adapt_filtered = np.zeros(6)
    
    def update(self, state, u_mpc, dt):
        """
        Update the adaptive control signal.
        
        Inputs:
          state : measured state (12D vector)
          u_mpc : baseline control from the MPC (6D vector)
          dt    : simulation/integration time step
          
        Returns:
          u_adapt : adaptive control signal (6D) to be added to u_mpc.
        """
        # Initialize the state predictor on the first call.
        if self.x_hat is None:
            self.x_hat = state.copy()
        
        # --- Adaptation law update:
        # Compute prediction error.
        e = state - self.x_hat
        # Map the 12D state error into the 6D control (matched) channel via B_pinv.
        delta = self.B_pinv @ e  # 6D vector
        
        # Define channel-specific adaptation gains as a diagonal (vector) multiplier.
        gains = np.array([
            self.gain_pos_horizontal,  # first horizontal force channel
            self.gain_pos_horizontal,  # second horizontal force channel
            self.gain_pos_vertical,    # vertical force channel
            self.gain_att,             # attitude channel 1
            self.gain_att,             # attitude channel 2
            self.gain_att              # attitude channel 3
        ])
        # Update the estimated uncertainty.
        self.adaptation_estimate += dt * gains * delta
        
        # Apply projection to ensure the estimated uncertainty stays within bounds.
        self.adaptation_estimate = np.clip(self.adaptation_estimate, -self.sigma_max, self.sigma_max)
        
        # --- State predictor update:
        # Predict the state derivative at x_hat using the nominal dynamics.
        f_nom = np.array(self.f_nominal(self.x_hat, u_mpc)).flatten()
        # The predictor dynamics include the estimated uncertainty.
        self.x_hat = self.x_hat + dt * (f_nom + self.B @ self.adaptation_estimate)
        
        # --- Adaptive control signal computation:
        # The raw adaptive control is chosen as the negative of the estimated uncertainty.
        raw_u_adapt = -self.adaptation_estimate
        
        # Apply separate first-order low-pass filters for translational and rotational channels.
        # For a discrete first-order filter: u_filtered = (1 - alpha)*u_filtered_prev + alpha*raw_u_adapt,
        # with alpha = dt/(dt + tau)
        # Translational channels (indices 0,1,2):
        alpha_trans = dt / (dt + self.tau_trans)
        self.u_adapt_filtered[0:3] = ((1 - alpha_trans) * self.u_adapt_filtered[0:3] + alpha_trans * raw_u_adapt[0:3])
        # Rotational channels (indices 3,4,5):
        alpha_rot = dt / (dt + self.tau_rot)
        self.u_adapt_filtered[3:6] = ((1 - alpha_rot) * self.u_adapt_filtered[3:6] + alpha_rot * raw_u_adapt[3:6])
        
        # --- (Optional) Additional scaling:
        # Here, we apply the same channel-specific gains to the filtered control.
        u_adapt = np.zeros(6)
        u_adapt[0:2] = self.gain_pos_horizontal * self.u_adapt_filtered[0:2]
        u_adapt[2]   = self.gain_pos_vertical   * self.u_adapt_filtered[2]
        u_adapt[3:6] = self.gain_att            * self.u_adapt_filtered[3:6]
        
        return u_adapt
    
    def reset(self, state=None):
        """
        Reset the internal state of the adaptive controller.
        
        This resets:
          - The state predictor (x_hat): If a 'state' is provided, it is used to reinitialize x_hat.
            Otherwise, x_hat is set to None and will be reinitialized during the next update.
          - The adaptation estimate (estimated uncertainty) is set to zero.
          - The filtered adaptive control signal is set to zero.
          
        Args:
            state (np.array, optional): Measured state (12D vector) to initialize the predictor.
        """
        self.x_hat = state.copy() if state is not None else None
        self.adaptation_estimate = np.zeros(6)
        self.u_adapt_filtered = np.zeros(6)
