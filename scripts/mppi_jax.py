import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

class Config:
    def __init__(self, 
                 T=10,  # Horizon (s)
                 dt=0.1,  # Length of each step (s)
                 num_control_rollouts=1024,  # Number of control sequences
                 num_controls=6,
                 num_states=12,
                 num_vis_state_rollouts=20,  # Number of visualization rollouts
                 seed=1):
        self.seed = seed
        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt)
        self.num_controls = num_controls
        self.num_states = num_states
        self.num_control_rollouts = num_control_rollouts
        self.num_vis_state_rollouts = num_vis_state_rollouts

DEFAULT_OBS_COST = 1e3
DEFAULT_DIST_WEIGHT = 10

# Weights for cost terms
STAGE_COST_WEIGHTS = jnp.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0], dtype=jnp.float32)
TERMINAL_COST_WEIGHTS = jnp.array([1000, 1000, 2000, 0, 0, 0, 5000, 5000, 10000, 0, 0, 0], dtype=jnp.float32)

CONTACT_NORMAL = jnp.array([-1, 0, 0], dtype=jnp.float32)


def dynamics_update_sim(x, u, dt):
    # The dynamics update for hexarotor
    I_xx, I_yy, I_zz = 0.115125971, 0.116524229, 0.230387752
    mass = 2.57
    g = 9.81
    
    x_next = x.copy()

    x_next = x_next.at[0].set(x[0] + dt * x[3])
    x_next = x_next.at[1].set(x[1] + dt * x[4])
    x_next = x_next.at[2].set(x[2] + dt * x[5])
  
    x_next = x_next.at[3].set(x[3] + dt * ((1/mass) * u[0] + g * jnp.sin(x[7])))
    x_next = x_next.at[4].set(x[4] + dt * ((1/mass) * u[1] - g * jnp.cos(x[7]) * jnp.sin(x[6])))
    x_next = x_next.at[5].set(x[5] + dt * ((1/mass) * u[2] - g * jnp.cos(x[7]) * jnp.cos(x[6])))

    x_next = x_next.at[6].set(x[6] + dt * (x[9] + x[10] * jnp.sin(x[6]) * jnp.tan(x[7]) + x[11] * jnp.cos(x[6]) * jnp.tan(x[7])))
    x_next = x_next.at[7].set(x[7] + dt * (x[10] * jnp.cos(x[6]) - x[11] * jnp.sin(x[6])))
    x_next = x_next.at[8].set(x[8] + dt * (x[10] * jnp.sin(x[6]) / jnp.cos(x[7]) + x[11] * jnp.cos(x[6]) / jnp.cos(x[7])))

    x_next = x_next.at[9].set(x[9] + dt * (1 / I_xx) * (u[3] + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
    x_next = x_next.at[10].set(x[10] + dt * (1 / I_yy) * (u[4] - I_xx * x[9] * x[11] + I_zz * x[9] * x[11]))
    x_next = x_next.at[11].set(x[11] + dt * (1 / I_zz) * (u[5] + I_xx * x[9] * x[10] - I_yy * x[9] * x[10]))

    return x_next


@jax.jit
def dynamics_update(x, u, dt, contact_normal):
    I_xx, I_yy, I_zz = 0.115125971, 0.116524229, 0.230387752
    mass = 2.57
    contact_normal_sq = 1
    fx_total, fy_total, fz_total = u[0], u[1], u[2]
    mx_total, my_total, mz_total = u[3], u[4], u[5]

    x = x.at[0].set(x[0] + dt * x[3])
    x = x.at[1].set(x[1] + dt * x[4])
    x = x.at[2].set(x[2] + dt * x[5])

    x = x.at[3].set(x[3] + dt * (1 / mass) * fx_total)
    x = x.at[4].set(x[4] + dt * (1 / mass) * fy_total)
    x = x.at[5].set(x[5] + dt * (1 / mass) * fz_total)

    x = x.at[6].set(x[6] + dt * (x[9] + x[10] * jnp.sin(x[6]) * jnp.tan(x[7]) + x[11] * jnp.cos(x[6]) * jnp.tan(x[7])))
    x = x.at[7].set(x[7] + dt * (x[10] * jnp.cos(x[6]) - x[11] * jnp.sin(x[6])))
    x = x.at[8].set(x[8] + dt * (x[10] * jnp.sin(x[6]) / jnp.cos(x[7]) + x[11] * jnp.cos(x[6]) / jnp.cos(x[7])))

    x = x.at[9].set(x[9] + dt * (1 / I_xx) * (mx_total + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
    x = x.at[10].set(x[10] + dt * (1 / I_yy) * (my_total - I_xx * x[9] * x[11] + I_zz * x[9] * x[11]))
    x = x.at[11].set(x[11] + dt * (1 / I_zz) * (mz_total + I_xx * x[9] * x[10] - I_yy * x[9] * x[10]))

    return x


@jax.jit
def stage_cost(dist2, dist_weight):
    return dist_weight * dist2


@jax.jit
def term_cost(dist2, goal_reached):
    return (1 - jnp.float32(goal_reached)) * dist2 * 10000
class MPPI_JAX:
    def __init__(self, cfg):
        self.cfg = cfg
        self.T = cfg.T
        self.dt = cfg.dt
        self.num_steps = cfg.num_steps
        self.num_control_rollouts = cfg.num_control_rollouts
        self.num_controls = cfg.num_controls
        self.num_states = cfg.num_states
        self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
        self.seed = cfg.seed

        self.u_seq0 = jnp.zeros((self.num_steps, self.num_controls))
        self.last_controls = jnp.zeros((self.num_control_rollouts, self.num_steps, self.num_controls))
        self.key = jax.random.PRNGKey(self.seed)
        self.device_var_initialized = False
        self.reset()

    def reset(self):
        self.u_seq0 = jnp.zeros((self.num_steps, self.num_controls))
        self.params = None
        self.params_set = False
        self.last_noise_d = jnp.zeros((self.num_control_rollouts, self.num_steps, self.num_controls))
        self.init_device_vars_before_solving()

    def init_device_vars_before_solving(self):
        """Initialize all necessary device variables before solving."""
        if not self.device_var_initialized:
            print("Initializing device variables...")
            # Initialize the variables such as noise samples, controls, etc.
            self.noise_samples_d = jnp.zeros((self.num_control_rollouts, self.num_steps, self.num_controls))
            self.u_cur_d = jnp.zeros((self.num_steps, self.num_controls))
            self.u_prev_d = jnp.zeros((self.num_steps, self.num_controls))
            self.costs_d = jnp.zeros((self.num_control_rollouts,))
            self.weights_d = jnp.zeros((self.num_control_rollouts,))
            # Any additional device variables initialization...
            self.device_var_initialized = True
            print("Device variables initialized.")
            self.device_var_initialized = True

    
    def set_params(self, params):
        self.params = params
        self.params_set = True

    def check_solve_conditions(self):
        if not self.params_set:
            print("MPPI parameters are not set. Cannot solve.")
            return False
        if not self.device_var_initialized:
            print("Device variables not initialized. Cannot solve.")
            return False
        return True

    def solve(self):
        
        if not self.check_solve_conditions():
            return
        return self.solve_with_nominal_dynamics()

    def move_mppi_task_vars_to_device(self):
        self.xgoal_d = self.params['xgoal']
        self.vrange_d = self.params['vrange']
        self.wrange_d = self.params['wrange']
        self.u_std_d = self.params['u_std']
        self.x0_d = self.params['x0']
        self.dt_d = self.dt

    def solve_with_nominal_dynamics(self):
        """
        Launch GPU kernels that use nominal dynamics but adjust cost function based on worst-case linear speed.
        """
        self.move_mppi_task_vars_to_device()
        for k in range(self.params['num_opt']):
            # Sample control noise
            self.sample_noise()

            # Rollout and compute mean or CVaR
            self.rollout_numba()
            # Compute cost and update the optimal control on the device
            self.update_useq_numba()

    def sample_noise(self):
        self.key, subkey = jax.random.split(self.key)
        u_std = self.u_std_d
        self.noise_samples_d = jax.random.normal(subkey, (self.num_control_rollouts, self.num_steps, self.num_controls)) * u_std

    def rollout_numba(self):
        def single_rollout(u_seq):
            x = self.x0_d
            total_cost = 0.0
            for t in range(self.num_steps):
                x = dynamics_update(x, u_seq[t], self.dt_d, CONTACT_NORMAL)
                dist2 = jnp.sum((self.xgoal_d[:3] - x[:3]) ** 2)
                total_cost += stage_cost(dist2, DEFAULT_DIST_WEIGHT)
            return total_cost

        self.costs_d = jax.vmap(single_rollout)(self.last_controls)

    def update_useq_numba(self):
        """
        Update the optimal control sequence based on previously evaluated cost values.
        """
        # Compute minimum cost
        beta = jnp.min(self.costs_d)

        # Compute weights using the formula: exp(-1/lambda * (costs - beta))
        lambda_weight_d = self.params['lambda_weight']
        weights = jnp.exp(-1.0 / lambda_weight_d * (self.costs_d - beta))

        # Normalize the weights
        weights /= jnp.sum(weights)

        # Update the optimal control sequence using the weighted sum of noise samples
        def update_control(t):
            return jnp.sum(weights[:, None] * self.noise_samples_d[:, t, :], axis=0)

        self.u_cur_d = jax.vmap(update_control)(jnp.arange(self.num_steps))

        # Apply control limits
        self.u_cur_d = jnp.clip(self.u_cur_d, self.vrange_d[0], self.vrange_d[1])

    def shift_and_update(self, new_x0, u_cur, num_shifts=1):
        self.params["x0"] = new_x0
        self.u_seq0 = jnp.roll(u_cur, shift=-num_shifts, axis=0)

    def get_state_rollout(self):
        def single_rollout(u_seq):
            x = self.params['x0']
            rollout_states = [x]
            for t in range(self.num_steps):
                x = dynamics_update(x, u_seq[t], self.dt_d, CONTACT_NORMAL)
                rollout_states.append(x)
            return jnp.array(rollout_states)

        self.state_rollout_batch_d = jax.vmap(single_rollout)(self.last_controls[:self.num_vis_state_rollouts])
        return self.state_rollout_batch_d


if __name__ == "__main__":
    # Configuration
    num_controls = 6
    num_states = 12
    cfg = Config(
        T=1.0,  # Horizon length in seconds
        dt=0.02,  # Time step
        num_control_rollouts=1024,  # Number of control sequences to sample
        num_controls=num_controls,
        num_states=num_states,  # Dimensionality of system states
        num_vis_state_rollouts=1,  # For visualization purposes
        seed=1
    )

    # Initial state and goal
    x0 = jnp.zeros(num_states)
    xgoal = jnp.array([1, -1, 3, 0, 0, 0, 0.1, -0.1, 0.3, 0, 0, 0])

    # MPPI Parameters
    mppi_params = dict(
        dt=cfg.dt,
        x0=x0,  # Start state
        xgoal=xgoal,  # Goal position
        goal_tolerance=0.001,
        dist_weight=2000,  # Weight for dist-to-goal cost
        lambda_weight=20.0,  # Temperature parameter in MPPI
        num_opt=5,  # Number of optimization steps in each solve() call
        u_std=jnp.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]) * 0.05,  # Noise standard deviation for controls
        vrange=jnp.array([-10.0, 10.0]),  # Control limits for linear velocities
        wrange=jnp.array([-0.1, 0.1]),  # Control limits for angular velocities
    )

    # Initialize the MPPI Controller
    mppi_controller = MPPI_JAX(cfg)
    mppi_controller.set_params(mppi_params)

    # Initialize state and control history
    max_steps = 1000
    xhist = jnp.zeros((max_steps + 1, num_states)) * jnp.nan
    uhist = jnp.zeros((max_steps, num_controls)) * jnp.nan
    xhist = xhist.at[0].set(x0)

    # Main Loop
    for t in range(max_steps):
        # Solve to get optimal control sequence
        print("timestep: ", t)
        u_seq = mppi_controller.solve()
        print(u_seq)
        u_curr = u_seq[0]

        # Compute gravity compensation in the body frame
        phi, theta, psi = xhist[t, 6:9]
        gravity_vector_world = jnp.array([0, 0, 9.81 * 2.57])
        R = jnp.array([
            [jnp.cos(theta) * jnp.cos(psi),
             jnp.sin(phi) * jnp.sin(theta) * jnp.cos(psi) - jnp.cos(phi) * jnp.sin(psi),
             jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi) + jnp.sin(phi) * jnp.sin(psi)],
            [jnp.cos(theta) * jnp.sin(psi),
             jnp.sin(phi) * jnp.sin(theta) * jnp.sin(psi) + jnp.cos(phi) * jnp.cos(psi),
             jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi) - jnp.sin(phi) * jnp.cos(psi)],
            [-jnp.sin(theta),
             jnp.sin(phi) * jnp.cos(theta),
             jnp.cos(phi) * jnp.cos(theta)]
        ])
        gravity_body = R.T @ gravity_vector_world
        u_curr = u_curr.at[:3].add(gravity_body)

        # Store current control
        uhist = uhist.at[t].set(u_curr)

        # Simulate the state forward
        xhist = xhist.at[t + 1].set(dynamics_update_sim(xhist[t], u_curr, cfg.dt))

        # Add some noise to the current state and update MPPI
        x_current_noisy = xhist[t + 1]
        x_current_noisy = x_current_noisy.at[:3].add(0.01 * jnp.random.randn(3))
        mppi_controller.shift_and_update(x_current_noisy, u_seq, num_shifts=1)

    # Plotting Results
    fig, axs = plt.subplots(4, 3, figsize=(12, 9))

    x_goal, y_goal, z_goal = xgoal[:3]
    roll_goal, pitch_goal, yaw_goal = xgoal[6:9]

    # Plot X, Y, Z positions
    axs[0][0].plot(xhist[:, 0], label='x')
    axs[0][0].axhline(x_goal, color='green', linestyle='--', label='X Goal')
    axs[0][0].set_title('X')
    axs[0][0].legend()

    axs[0][1].plot(xhist[:, 1], label='y')
    axs[0][1].axhline(y_goal, color='green', linestyle='--', label='Y Goal')
    axs[0][1].set_title('Y')
    axs[0][1].legend()

    axs[0][2].plot(xhist[:, 2], label='z')
    axs[0][2].axhline(z_goal, color='green', linestyle='--', label='Z Goal')
    axs[0][2].set_title('Z')
    axs[0][2].legend()

    # Plot roll, pitch, yaw
    axs[1][0].plot(xhist[:, 6] * 180 / jnp.pi, label='roll')
    axs[1][0].axhline(roll_goal * 180 / jnp.pi, color='green', linestyle='--', label='Roll Goal')
    axs[1][0].set_title('Roll')
    axs[1][0].legend()

    axs[1][1].plot(xhist[:, 7] * 180 / jnp.pi, label='pitch')
    axs[1][1].axhline(pitch_goal * 180 / jnp.pi, color='green', linestyle='--', label='Pitch Goal')
    axs[1][1].set_title('Pitch')
    axs[1][1].legend()

    axs[1][2].plot(xhist[:, 8] * 180 / jnp.pi, label='yaw')
    axs[1][2].axhline(yaw_goal * 180 / jnp.pi, color='green', linestyle='--', label='Yaw Goal')
    axs[1][2].set_title('Yaw')
    axs[1][2].legend()

    # Plot control inputs
    axs[2][0].plot(uhist[:, 0], label='Fx')
    axs[2][0].set_title('Control Fx')
    axs[2][0].legend()

    axs[2][1].plot(uhist[:, 1], label='Fy')
    axs[2][1].set_title('Control Fy')
    axs[2][1].legend()

    axs[2][2].plot(uhist[:, 2], label='Fz')
    axs[2][2].set_title('Control Fz')
    axs[2][2].legend()

    axs[3][0].plot(uhist[:, 3], label='Mx')
    axs[3][0].set_title('Control Mx')
    axs[3][0].legend()

    axs[3][1].plot(uhist[:, 4], label='My')
    axs[3][1].set_title('Control My')
    axs[3][1].legend()

    axs[3][2].plot(uhist[:, 5], label='Mz')
    axs[3][2].set_title('Control Mz')
    axs[3][2].legend()

    plt.tight_layout()
    plt.show()