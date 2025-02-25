import numpy as np
import casadi as cs
from scipy.signal import savgol_filter  # Requires SciPy
from nav_msgs.msg import Odometry
import rospy

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
        
        self.fmax = params.get("max_force", 10)
        self.mmax = params.get("max_torque", 0.1)
        self.umax = np.array([self.fmax, self.fmax, 8*self.fmax, self.mmax, self.mmax, self.mmax])
        self.umin = np.array([-self.fmax, -self.fmax, 0       , -self.mmax, -self.mmax, -self.mmax])
        
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
        self.n = 12
        self.z_dim = 2 * self.n
        self.ekf_state = np.zeros(self.z_dim)  # Initially, x_est = 0 and d_est = 0.
        self.P = np.eye(self.z_dim) * 1e-3    # Initial covariance.
        # Process noise covariance Q (can be tuned).
        self.Q = np.diag([
            # Physical states (0-11)
            1e-2, 1e-2, 1e-2,   # x,y,z position
            1e-1, 1e-1, 1e-1,   # vx,vy,vz velocity  
            1e-5, 1e-5, 1e-5,   # φ,θ,ψ attitude  
            1e-4, 1e-4, 1e-4,   # p,q,r angular rates
            
            # Disturbance states (12-23)
            1e-3, 1e-3, 5e-3,   # x,y,z position derivatives (unmatched)
            1e-1, 1e-1, 5e-1,   # vx,vy,vz acceleration (matched - high Q)
            1e-4, 1e-4, 1e-4,   # φ,θ,ψ angular rate derivatives (unmatched)  
            1e-1, 1e-1, 1e-1    # p,q,r angular acceleration (matched - high Q)
        ])
        # Measurement noise covariance (R)
        self.R = np.diag([1e-3, 1e-3, 2e-3] +    # Position (x,y,z)
                        [1e-2, 1e-2, 5e-3] +    # Velocity (vx,vy,vz)
                        [1e-5, 1e-5, 1e-5] +    # Attitude
                        [1e-4, 1e-4, 1e-4])     # Angular rates

        # --- Add a forgetting factor to the disturbance estimate (to prevent drift).
        self.disturbance_forgetting_rate = params.get("l1_disturbance_forgetting_rate", 0.001)

        # Tuning debug publishers
        debug_topics = [
            "/mppi/l1_ekf_debug/innovation",
            "/mppi/l1_ekf_debug/disturbance",
            "/mppi/l1_ekf_debug/covariance"
        ]
        self.debug_publishers = {}
        for topic in debug_topics:
            self.debug_publishers[topic] = rospy.Publisher(topic, Odometry, queue_size=10)

    def _ekf_update(self, measurement, u, dt):
        """
        EKF prediction and update for the augmented state z = [x; d].
        Process model:
          x_{k+1} = x_k + dt*( f_nominal(x_k, u) + d_k )
          d_{k+1} = d_k
        Measurement model:
          y = x
        """
        n = self.n
        z = self.ekf_state  # z = [x; d]
        x = z[:n]
        d = z[n:]
        
        # Predict step.
        f_val = np.array(self.f_nominal(x, u)).flatten()  # f_nominal(x,u)
        x_pred = x + dt * (f_val + d)
        d_pred = d  # d is assumed constant.
        z_pred = np.concatenate([x_pred, d_pred])
        
        # Compute the Jacobian F = dF/dz.
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

        self.publish_debug_odometry(innovation, "/mppi/l1_ekf_debug/innovation")
        
        # Update step.
        z_upd = z_pred + K @ innovation
        P_upd = (np.eye(self.z_dim) - K @ H) @ P_pred
        
        # Apply a forgetting factor to the disturbance estimate to prevent drift.
        d_est = z_upd[n:]
        d_est = (1 - self.disturbance_forgetting_rate) * d_est
        z_upd[n:] = d_est

        # Save updated state.
        self.ekf_state = z_upd
        self.P = P_upd

        P_debug = np.diag(self.P)
        self.publish_debug_odometry(P_debug, "/mppi/l1_ekf_debug/covariance")
        
        return d_est

    def update(self, state, u_mpc, dt):
        """
        Update the adaptive control signal using the EKF estimate of the disturbance.
        """
        # Get the disturbance estimate from the EKF.
        d_hat = self._ekf_update(measurement=state, u=u_mpc, dt=dt)
        self.publish_debug_odometry(d_hat, "/mppi/l1_ekf_debug/disturbance")
        
        # Project the disturbance (in state derivative space) into the control space.
        u_dist_est = self.B_pinv @ d_hat

        # Add the current u_dist_est to the Savitzky–Golay filter buffer.
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
        alpha_trans = dt / (dt + self.tau_trans)
        self.adapted_control[0:3] = ((1 - alpha_trans) * self.adapted_control[0:3] +
                                     alpha_trans * u_dist_est_filtered[0:3])
        alpha_rot = dt / (dt + self.tau_rot)
        self.adapted_control[3:6] = ((1 - alpha_rot) * self.adapted_control[3:6] +
                                     alpha_rot * u_dist_est_filtered[3:6])
        
        # Apply separate adaptation gains.
        u_adapt = np.zeros(6)
        u_adapt[0:2] = self.adaptation_gain_pos_horizontal * self.adapted_control[0:2]
        u_adapt[2]   = self.adaptation_gain_pos_vertical * self.adapted_control[2]
        u_adapt[3:6] = self.adaptation_gain_att * self.adapted_control[3:6]

        # Apply a deadzone threshold to avoid amplifying very small noise values.
        deadzone_threshold = 1e-3
        u_adapt[np.abs(u_adapt) < deadzone_threshold] = 0.0

        # Clip the adaptive control signal to the actuator limits.
        u_adapt = np.clip(u_adapt, self.umin, self.umax)

        return u_adapt
    
    def publish_debug_odometry(self, state, topic):
        """
        Publish debug information about the state and covariance.
        """
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()

        # Set the position.
        odom_msg.pose.pose.position.x = state[0]
        odom_msg.pose.pose.position.y = state[1]
        odom_msg.pose.pose.position.z = state[2]
        # Set the velocity.
        odom_msg.twist.twist.linear.x = state[3]
        odom_msg.twist.twist.linear.y = state[4]
        odom_msg.twist.twist.linear.z = state[5]
        # Set the covariance (here simply a placeholder).
        odom_msg.pose.covariance = [0.1] * 36
        odom_msg.twist.covariance = [0.1] * 36

        # Set the orientation.
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.pose.pose.orientation.x = state[6]
        odom_msg.pose.pose.orientation.y = state[7]
        odom_msg.pose.pose.orientation.z = state[8]

        # Set the angular velocity.
        odom_msg.twist.twist.angular.x = state[9]
        odom_msg.twist.twist.angular.y = state[10]
        odom_msg.twist.twist.angular.z = state[11]

        # Create the publisher if it doesn't exist.
        if topic not in self.debug_publishers:
            self.debug_publishers[topic] = rospy.Publisher(topic, Odometry, queue_size=10)
        else:
            self.debug_publishers[topic].publish(odom_msg)

    def reset(self, state=None):
        """
        Reset the internal state of the adaptive controller.
        """
        self.adapted_control = np.zeros(6)
        self.disturbance_buffer = []
        if state is not None:
            self.ekf_state = np.concatenate([state, np.zeros(self.n)])
        else:
            self.ekf_state = np.zeros(self.z_dim)
        self.P = np.eye(self.z_dim) * 1e-3
