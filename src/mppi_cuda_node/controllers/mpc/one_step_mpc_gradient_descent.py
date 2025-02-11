import numpy as np
import math

class OneStepMPC:
    def __init__(self, params):
        self.params = params
        self.previous_control = np.zeros(6)  # Store previous control for smoothness

    def compute_control(self, state, target_state, initial_guess, dt, max_iters=50, alpha=0.01):
        # Extract parameters
        mass = self.params['mass']
        g = self.params['gravity']
        I_xx, I_yy, I_zz = self.params['inertia']
        max_force = self.params['max_force']
        max_torque = self.params['max_torque']
        Q = np.diag([self.params['tracking_weight_pos']] * 3 + [self.params['tracking_weight_vel']] * 3 +
                    [self.params['tracking_weight_att']] * 3 + [self.params['tracking_weight_ang_vel']] * 3)
        R = np.eye(6) * self.params['control_weight']
        S = np.eye(6) * self.params['smoothness_weight']

        def dynamics_update(x, u):
            """Compute next state using Euler integration of the full dynamics."""
            dx = np.zeros(12)
            dx[0] = x[3]
            dx[1] = x[4]
            dx[2] = x[5]
            dx[3] = (1 / mass) * u[0] - g * (math.cos(x[6]) * math.sin(x[7]) * math.cos(x[8]) + math.sin(x[6]) * math.sin(x[8]))
            dx[4] = (1 / mass) * u[1] - g * (math.cos(x[6]) * math.sin(x[7]) * math.sin(x[8]) - math.sin(x[6]) * math.cos(x[8]))
            dx[5] = (1 / mass) * u[2] - g * (math.cos(x[6]) * math.cos(x[7]))
            dx[6] = x[9] + x[10] * math.sin(x[6]) * math.tan(x[7]) + x[11] * math.cos(x[6]) * math.tan(x[7])
            dx[7] = x[10] * math.cos(x[6]) - x[11] * math.sin(x[6])
            dx[8] = x[10] * math.sin(x[6]) / math.cos(x[7]) + x[11] * math.cos(x[6]) / math.cos(x[7])
            dx[9] = (1 / I_xx) * (u[3] + I_yy * x[10] * x[11] - I_zz * x[10] * x[11])
            dx[10] = (1 / I_yy) * (u[4] - I_xx * x[9] * x[11] + I_zz * x[9] * x[11])
            dx[11] = (1 / I_zz) * (u[5] + I_xx * x[9] * x[10] - I_yy * x[9] * x[10])
            return x + dx * dt

        def cost(u):
            """Compute the total cost."""
            x_next = dynamics_update(state, u)
            tracking_error = np.dot((x_next - target_state).T, Q @ (x_next - target_state))
            control_effort = np.dot(u.T, R @ u)
            smoothness_penalty = np.dot((u - self.previous_control).T, S @ (u - self.previous_control))
            return tracking_error + control_effort + smoothness_penalty

        def gradient(u):
            """Compute the gradient of the cost with respect to u."""
            eps = 1e-5  # Small perturbation for numerical gradient
            grad = np.zeros_like(u)
            for i in range(len(u)):
                u_perturb = u.copy()
                u_perturb[i] += eps
                cost_plus = cost(u_perturb)
                u_perturb[i] -= 2 * eps
                cost_minus = cost(u_perturb)
                grad[i] = (cost_plus - cost_minus) / (2 * eps)
            return grad

        # Gradient descent loop
        u = initial_guess.copy()
        for _ in range(max_iters):
            grad = gradient(u)
            u -= alpha * grad
            # Project onto bounds
            u[:3] = np.clip(u[:3], -max_force, max_force)
            u[3:] = np.clip(u[3:], -max_torque, max_torque)

        # Update previous control
        self.previous_control = u
        return u

if __name__ == '__main__':
    # Example usage
    params = {
        'inertia': [0.05132, 0.06282, 0.06248],
        'mass': 3.49,
        'gravity': 9.81,
        'max_force': 20.0,
        'max_torque': 0.05,
        'control_weight': 0.01,
        'tracking_weight_pos': 1,
        'tracking_weight_vel': 0.1,
        'tracking_weight_att': 1,
        'tracking_weight_ang_vel': 0.5,
        'smoothness_weight': 0.01,
    }

    mpc = GradientDescentMPC(params)
    state = np.zeros(12)
    target_state = np.array([1.0, 1.0, 2.0] + [0.0] * 9)
    initial_guess = np.zeros(6)

    control = mpc.compute_control(state, target_state, initial_guess, dt=0.02)
    print("Control:", control)
