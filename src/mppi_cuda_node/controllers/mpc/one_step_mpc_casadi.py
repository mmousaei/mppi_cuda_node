import casadi as ca
import numpy as np

class OneStepMPC:
    def __init__(self, params):
        """
        Initialize the one-step MPC controller.

        Parameters:
        params: Dictionary containing system parameters such as inertia, mass, gravity, etc.
        """
        self.params = params
        self.previous_control_position = np.zeros(3)  # Store previous position control inputs
        self.previous_control_orientation = np.zeros(3)  # Store previous orientation control inputs

    def compute_position_control(self, state, target_state, initial_guess, dt):
        """
        Compute the optimal control input for position.
        """
        mass = self.params['mass']
        g = self.params['gravity']
        max_force = self.params['max_force']
        control_weight = self.params['control_weight']
        tracking_weight_pos = self.params['tracking_weight_pos']
        tracking_weight_vel = self.params['tracking_weight_vel']

        # Decision variables
        u = ca.SX.sym('u', 3)  # Control inputs (forces)

        # Define the position dynamics
        def dynamics_update(x, u):
            dx = ca.SX.zeros(6)
            dx[0:3] = x[3:6]
            dx[3] = (1 / mass) * u[0]
            dx[4] = (1 / mass) * u[1]
            dx[5] = (1 / mass) * u[2] - g
            return x + dx * dt

        # Compute the next state
        next_state = dynamics_update(state[:6], u)

        # Define the cost function
        tracking_error_pos = ca.sumsqr(next_state[:3] - target_state[:3])
        tracking_error_vel = ca.sumsqr(next_state[3:6] - target_state[3:6])
        control_effort = ca.sumsqr(u)

        cost = (
            tracking_weight_pos * tracking_error_pos +
            tracking_weight_vel * tracking_error_vel +
            control_weight * control_effort
        )

        # Optimization problem
        nlp = {'x': u, 'f': cost}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})

        # Solve the optimization problem
        bounds = {'lbx': [-max_force, -max_force, 0], 'ubx': [max_force, max_force, max_force]}
        solution = solver(x0=initial_guess[:3], lbx=bounds['lbx'], ubx=bounds['ubx'])
        optimal_control = solution['x'].full().flatten()

        # Update previous control input
        self.previous_control_position = optimal_control

        return optimal_control

    def compute_orientation_control(self, state, target_state, initial_guess, dt):
        """
        Compute the optimal control input for orientation.
        """
        I_xx, I_yy, I_zz = self.params['inertia']
        max_torque = self.params['max_torque']
        control_weight = self.params['control_weight']
        tracking_weight_att = self.params['tracking_weight_att']
        tracking_weight_ang_vel = self.params['tracking_weight_ang_vel']

        # Decision variables
        u = ca.SX.sym('u', 3)  # Control inputs (torques)

        # Define the orientation dynamics
        def dynamics_update(x, u):
            dx = ca.SX.zeros(6)
            dx[0:3] = x[3:6]
            dx[3] = (1 / I_xx) * u[0]
            dx[4] = (1 / I_yy) * u[1]
            dx[5] = (1 / I_zz) * u[2]
            return x + dx * dt

        # Compute the next state
        next_state = dynamics_update(state[6:], u)

        # Define the cost function
        tracking_error_att = ca.sumsqr(next_state[:3] - target_state[6:9])
        tracking_error_ang_vel = ca.sumsqr(next_state[3:6] - target_state[9:12])
        control_effort = ca.sumsqr(u)

        cost = (
            tracking_weight_att * tracking_error_att +
            tracking_weight_ang_vel * tracking_error_ang_vel +
            control_weight * control_effort
        )

        # Optimization problem
        nlp = {'x': u, 'f': cost}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})

        # Solve the optimization problem
        bounds = {'lbx': [-max_torque, -max_torque, -max_torque], 'ubx': [max_torque, max_torque, max_torque]}
        solution = solver(x0=initial_guess[3:], lbx=bounds['lbx'], ubx=bounds['ubx'])
        optimal_control = solution['x'].full().flatten()

        # Update previous control input
        self.previous_control_orientation = optimal_control

        return optimal_control

    def compute_control(self, state, target_state, initial_guess, dt):
        """
        Compute the overall control by solving position and orientation problems separately.
        """
        position_control = self.compute_position_control(state, target_state, initial_guess, dt)
        orientation_control = self.compute_orientation_control(state, target_state, initial_guess, dt)
        return np.concatenate([position_control, orientation_control])

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
