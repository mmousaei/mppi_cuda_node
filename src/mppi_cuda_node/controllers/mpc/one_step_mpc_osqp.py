import osqp
import numpy as np
from scipy.sparse import csc_matrix

class OneStepMPC:
    def __init__(self, params):
        """
        Initialize the one-step MPC controller using OSQP.

        Parameters:
        params: Dictionary containing system parameters such as inertia, mass, gravity, etc.
        """
        self.params = params
        self.previous_control = np.zeros(6)  # Store the previous control input for smoothness

    def compute_control(self, state, target_state, initial_guess, dt):
        """
        Compute the optimal control input for the next step using one-step MPC.

        Parameters:
        state: Current state of the system (12D vector).
        target_state: Desired state of the system (12D vector).
        initial_guess: Initial guess for the control input (6D vector).
        dt: Time step for prediction.

        Returns:
        Optimal control input (6D vector).
        """
        # Extract parameters
        I_xx, I_yy, I_zz = self.params['inertia']
        mass = self.params['mass']
        g = self.params['gravity']
        max_force = self.params['max_force']
        max_torque = self.params['max_torque']
        control_weight = self.params['control_weight']
        tracking_weight_pos = self.params['tracking_weight_pos']
        tracking_weight_vel = self.params['tracking_weight_vel']
        tracking_weight_att = self.params['tracking_weight_att']
        tracking_weight_ang_vel = self.params['tracking_weight_ang_vel']
        smoothness_weight = self.params['smoothness_weight']

        # Linearized dynamics around current state
        phi, theta, psi = state[6:9]  # Roll, pitch, yaw
        A = np.eye(12)
        B = np.zeros((12, 6))

        # Forces affecting position (x, y, z)
        B[3, 0] = 1 / mass
        B[4, 1] = 1 / mass
        B[5, 2] = 1 / mass

        # Gravity compensation in z-direction
        gravity_compensation = g * mass * np.cos(phi) * np.cos(theta)
        A[5, 5] += gravity_compensation * dt

        # Torques affecting angular velocities (roll, pitch, yaw rates)
        B[9, 3] = 1 / I_xx
        B[10, 4] = 1 / I_yy
        B[11, 5] = 1 / I_zz

        # Cost matrices
        Q = np.diag(
            [
                tracking_weight_pos, tracking_weight_pos, tracking_weight_pos,  # Position weights
                tracking_weight_vel, tracking_weight_vel, tracking_weight_vel,  # Velocity weights
                tracking_weight_att, tracking_weight_att, tracking_weight_att,  # Attitude weights
                tracking_weight_ang_vel, tracking_weight_ang_vel, tracking_weight_ang_vel,  # Angular velocity weights
            ]
        )

        R = np.diag(
            [control_weight, control_weight, control_weight, control_weight, control_weight, control_weight]
        )

        S = np.diag([smoothness_weight] * 6)

        # OSQP requires cost function in the form: 1/2 * u.T * P * u + q.T * u
        P = 2 * (B.T @ Q @ B + R + S)  # Quadratic cost matrix
        q = 2 * B.T @ Q @ (A @ state - target_state) + 2 * S @ (self.previous_control)

        # Constraints: lb <= A * u <= ub
        lb = np.array([-max_force, -max_force, -gravity_compensation, -max_torque, -max_torque, -max_torque])
        ub = np.array([max_force, max_force, max_force - gravity_compensation, max_torque, max_torque, max_torque])

        # Constraint matrix A (identity matrix for u)
        constraint_A = np.eye(6)

        # Convert matrices to sparse format for OSQP
        P_sparse = csc_matrix(P)
        constraint_A_sparse = csc_matrix(constraint_A)
        q = q.flatten()

        # OSQP Problem setup
        osqp_solver = osqp.OSQP()
        osqp_solver.setup(P=P_sparse, q=q, A=constraint_A_sparse, l=lb, u=ub, verbose=False)

        # Solve the optimization problem
        result = osqp_solver.solve()

        # Check solver status
        if result.info.status != "solved":
            raise RuntimeError("OSQP failed to solve the problem.")

        # Extract the optimal control input
        optimal_control = result.x[:6]

        # Update previous control input
        self.previous_control = optimal_control

        return optimal_control
    
if __name__ == '__main__':
    # Example usage
    params = {
        'inertia': [0.115125971, 0.116524229, 0.230387752],
        'mass': 3.49,
        'gravity': 9.81,
        'max_force': 60.0,
        'max_torque': 10.0,
        'control_weight': 0.01,
        'tracking_weight_pos': 1.0,
        'tracking_weight_vel': 0.5,
        'tracking_weight_att': 1.0,
        'tracking_weight_ang_vel': 0.5,
        'smoothness_weight': 0.1,
    }

    mpc = OneStepMPC(params)

    # Example states (12D)
    current_state = np.zeros(12)
    target_state = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    dt = 0.02  # 20ms timestep
    initial_guess = np.zeros(6)

    control_input = mpc.compute_control(current_state, target_state, initial_guess, dt)

    print("Optimal control input:", control_input)
