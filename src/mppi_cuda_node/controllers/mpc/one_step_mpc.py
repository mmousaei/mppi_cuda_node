import numpy as np
from scipy.optimize import minimize
import math

class OneStepMPC:
    def __init__(self, params):
        """
        Initialize the one-step MPC controller.

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

        def dynamics_update(x, u, dt):
            """
            RK4-based dynamics update for hexarotor.
            """
            I_xx, I_yy, I_zz = self.params['inertia']
            mass = self.params['mass']
            g = self.params['gravity']

            def compute_derivative(x_local, u_local):
                derivative = np.zeros(12)
                derivative[0] = x_local[3]
                derivative[1] = x_local[4]
                derivative[2] = x_local[5]
                derivative[3] = (1 / mass) * u_local[0] - g * (math.cos(x_local[6]) * math.sin(x_local[7]) * math.cos(x_local[8]) + math.sin(x_local[6]) * math.sin(x_local[8]))
                derivative[4] = (1 / mass) * u_local[1] - g * (math.cos(x_local[6]) * math.sin(x_local[7]) * math.sin(x_local[8]) - math.sin(x_local[6]) * math.cos(x_local[8]))
                derivative[5] = (1 / mass) * u_local[2] - g * (math.cos(x_local[6]) * math.cos(x_local[7]))
                derivative[6] = x_local[9] + x_local[10] * math.sin(x_local[6]) * math.tan(x_local[7]) + x_local[11] * math.cos(x_local[6]) * math.tan(x_local[7])
                derivative[7] = x_local[10] * math.cos(x_local[6]) - x_local[11] * math.sin(x_local[6])
                derivative[8] = x_local[10] * math.sin(x_local[6]) / math.cos(x_local[7]) + x_local[11] * math.cos(x_local[6]) / math.cos(x_local[7])
                derivative[9] = (1 / I_xx) * (u_local[3] + I_yy * x_local[10] * x_local[11] - I_zz * x_local[10] * x_local[11])
                derivative[10] = (1 / I_yy) * (u_local[4] - I_xx * x_local[9] * x_local[11] + I_zz * x_local[9] * x_local[11])
                derivative[11] = (1 / I_zz) * (u_local[5] + I_xx * x_local[9] * x_local[10] - I_yy * x_local[9] * x_local[10])
                return derivative

            k1 = compute_derivative(x, u) * dt
            k2 = compute_derivative(x + 0.5 * k1, u) * dt
            k3 = compute_derivative(x + 0.5 * k2, u) * dt
            k4 = compute_derivative(x + k3, u) * dt

            return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        def cost_function(control):
            """
            Compute the cost for a given control input.
            """
            control = np.array(control)

            # Predict the next state using RK4 dynamics
            next_state = dynamics_update(state, control, dt)

            # Tracking cost: penalize deviation from the target state
            tracking_error_pos = np.linalg.norm(next_state[:3] - target_state[:3]) ** 2
            tracking_error_vel = np.linalg.norm(next_state[3:6] - target_state[3:6]) ** 2
            tracking_error_att = np.linalg.norm(next_state[6:9] - target_state[6:9]) ** 2
            tracking_error_ang_vel = np.linalg.norm(next_state[9:12] - target_state[9:12]) ** 2

            # Control effort cost: penalize large control inputs
            control_effort = np.linalg.norm(control[:3]) ** 2 + 100*np.linalg.norm(control[3:]) ** 2

            # Smoothness cost: penalize changes in control inputs
            smoothness = np.linalg.norm(control - self.previous_control) ** 2

            total_cost = (
                tracking_weight_pos * tracking_error_pos +
                tracking_weight_vel * tracking_error_vel +
                tracking_weight_att * tracking_error_att +
                tracking_weight_ang_vel * tracking_error_ang_vel +
                control_weight * control_effort +
                smoothness_weight * smoothness
            )

            return total_cost

        # Constraints for control inputs
        bounds = [
            (-max_force, max_force),
            (-max_force, max_force),
            (0, max_force),  # Assume positive thrust only
            (-max_torque, max_torque),
            (-max_torque, max_torque),
            (-max_torque, max_torque),
        ]

        # Solve the optimization problem
        result = minimize(
            cost_function,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
        )

        # Update previous control input
        self.previous_control = result.x

        return result.x

if __name__ == '__main__':

    # Example usage
    params = {
        'inertia': [0.115125971, 0.116524229, 0.230387752],
        'mass': 3.49,
        'gravity': 9.81,
        'max_force': 60.0,
        'max_torque': 10.0,
        'control_weight': 0.01,
        'tracking_weight': 1.0,
        'smoothness_weight': 0.1,
    }

    mpc = OneStepMPC(params)

    # Example states (12D)
    current_state = np.zeros(12)
    target_state = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    dt = 0.02  # 20ms timestep
    control_input = mpc.compute_control(current_state, target_state, dt)

    print("Optimal control input:", control_input)
