import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
import matplotlib.pyplot as plt
import time

key = random.PRNGKey(0)

class Config:
    """Configurations that are typically fixed throughout execution."""
    def __init__(self, 
                 T=0.5,  # Horizon (s)
                 dt=0.02,  # Length of each step (s)
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

        # For visualizing state rollouts
        self.num_vis_state_rollouts = min(num_vis_state_rollouts, num_control_rollouts)
        self.num_vis_state_rollouts = max(1, self.num_vis_state_rollouts)

DEFAULT_OBS_COST = 1e3
DEFAULT_DIST_WEIGHT = 10

# Define stage and terminal cost weights for each state dimension
STAGE_COST_WEIGHTS = jnp.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0], dtype=jnp.float32)  # Example weights
TERMINAL_COST_WEIGHTS = jnp.array([1000, 1000, 2000, 0, 0, 0, 5000, 5000, 10000, 0, 0, 0], dtype=jnp.float32)  # Example weights

def dynamics_update_sim(x, u, dt):
    # The dynamics update for hexarotor (includes gravity)
    I_xx = 0.115125971
    I_yy = 0.116524229
    I_zz = 0.230387752
    mass = 2.302499999999999
    g = 9.81

    x_next = x.copy()

    x_next = x_next.at[0].add(dt * x[3])
    x_next = x_next.at[1].add(dt * x[4])
    x_next = x_next.at[2].add(dt * x[5])

    x_next = x_next.at[3].add(dt * ((1 / mass) * u[0] + g * jnp.sin(x[7])))
    x_next = x_next.at[4].add(dt * ((1 / mass) * u[1] - g * jnp.cos(x[7]) * jnp.sin(x[6])))
    x_next = x_next.at[5].add(dt * ((1 / mass) * u[2] - g * jnp.cos(x[7]) * jnp.cos(x[6])))

    x_next = x_next.at[6].add(dt * (x[9] + x[10] * jnp.sin(x[6]) * jnp.tan(x[7]) + x[11] * jnp.cos(x[6]) * jnp.tan(x[7])))
    x_next = x_next.at[7].add(dt * (x[10] * jnp.cos(x[6]) - x[11] * jnp.sin(x[6])))
    x_next = x_next.at[8].add(dt * (x[10] * jnp.sin(x[6]) / jnp.cos(x[7]) + x[11] * jnp.cos(x[6]) / jnp.cos(x[7])))

    x_next = x_next.at[9].add(dt * ((1 / I_xx) * (u[3] + (I_yy - I_zz) * x[10] * x[11])))
    x_next = x_next.at[10].add(dt * ((1 / I_yy) * (u[4] + (I_zz - I_xx) * x[9] * x[11])))
    x_next = x_next.at[11].add(dt * ((1 / I_zz) * (u[5] + (I_xx - I_yy) * x[9] * x[10])))

    return x_next

@jit
def dynamics_update(x, u, dt):
    # The dynamics update for hexarotor without gravity (for MPPI rollouts)
    I_xx = 0.115125971
    I_yy = 0.116524229
    I_zz = 0.230387752
    mass = 2.302499999999999

    x_next = x.copy()

    x_next = x_next.at[0].add(dt * x[3])
    x_next = x_next.at[1].add(dt * x[4])
    x_next = x_next.at[2].add(dt * x[5])

    x_next = x_next.at[3].add(dt * ((1 / mass) * u[0]))
    x_next = x_next.at[4].add(dt * ((1 / mass) * u[1]))
    x_next = x_next.at[5].add(dt * ((1 / mass) * u[2]))

    x_next = x_next.at[6].add(dt * (x[9] + x[10] * jnp.sin(x[6]) * jnp.tan(x[7]) + x[11] * jnp.cos(x[6]) * jnp.tan(x[7])))
    x_next = x_next.at[7].add(dt * (x[10] * jnp.cos(x[6]) - x[11] * jnp.sin(x[6])))
    x_next = x_next.at[8].add(dt * (x[10] * jnp.sin(x[6]) / jnp.cos(x[7]) + x[11] * jnp.cos(x[6]) / jnp.cos(x[7])))

    x_next = x_next.at[9].add(dt * ((1 / I_xx) * (u[3] + (I_yy - I_zz) * x[10] * x[11])))
    x_next = x_next.at[10].add(dt * ((1 / I_yy) * (u[4] + (I_zz - I_xx) * x[9] * x[11])))
    x_next = x_next.at[11].add(dt * ((1 / I_zz) * (u[5] + (I_xx - I_yy) * x[9] * x[10])))

    return x_next

# Stage cost function
@jit
def stage_cost(x, u, xgoal, dist_weight):
    # Compute the cost for a single timestep
    w_pose_xy = 1500
    w_pose_z =  1500
    w_vel = 250
    w_att = 28500
    w_omega = 1800
    w_cont = 5
    w_cont_m = 10
    w_cont_f = 1
    w_cont_M = 1

    pos_error = x[:3] - xgoal[:3]
    vel_error = x[3:6] - xgoal[3:6]
    att_error = x[6:9] - xgoal[6:9]
    omega_error = x[9:12] - xgoal[9:12]

    dist_to_goal2 = (
        w_pose_xy * (pos_error[0]**2 + 4 * pos_error[1]**2) + 
        w_pose_z * pos_error[2]**2 +
        w_vel * (vel_error[0]**2 + 4 * vel_error[1]**2 + vel_error[2]**2) +
        w_att * (att_error[0]**2 + 2 * att_error[1]**2 + 2 * att_error[2]**2) +
        w_omega * (omega_error[0]**2 + omega_error[1]**2 + 2 * omega_error[2]**2) +
        w_cont_f * (u[0]**2 + u[1]**2 + u[2]**2) +
        w_cont_M * (u[3]**2 + u[4]**2 + u[5]**2)
    )

    return dist_weight * dist_to_goal2

# Terminal cost function
@jit
def terminal_cost(x, xgoal, goal_tolerance, w_term):
    pos_error = x[:3] - xgoal[:3]
    dist_to_goal2 = jnp.sum(pos_error**2)
    goal_reached = dist_to_goal2 <= goal_tolerance**2
    return w_term * (1 - goal_reached) * dist_to_goal2

# Rollout function
def rollout(key, x0, u_seq, params):
    dt = params['dt']
    xgoal = params['xgoal']
    lambda_weight = params['lambda_weight']
    dist_weight = params['dist_weight']
    goal_tolerance = params['goal_tolerance']
    u_std = params['u_std']
    w_term = 1000

    def body_fun(carry, u):
        x, cost = carry
        key = random.fold_in(key, cost)
        noise = random.normal(key, shape=u.shape) * u_std
        u_noisy = u + noise
        x_next = dynamics_update(x, u_noisy, dt)
        c = stage_cost(x_next, u_noisy, xgoal, dist_weight)
        cost += c
        return (x_next, cost), None

    (x_final, total_cost), _ = jax.lax.scan(body_fun, (x0, 0.0), u_seq)
    total_cost += terminal_cost(x_final, xgoal, goal_tolerance, w_term)
    return total_cost

vmap_rollout = vmap(rollout, in_axes=(0, None, 0, None))

class MPPI_JAX:
    """MPPI controller implemented in JAX"""
    def __init__(self, cfg, key):
        self.cfg = cfg
        self.T = cfg.T
        self.dt = cfg.dt
        self.num_steps = cfg.num_steps
        self.num_control_rollouts = cfg.num_control_rollouts
        self.num_controls = cfg.num_controls
        self.num_states = cfg.num_states
        self.key = key

        # Initialize control sequences
        self.u_seq = jnp.zeros((self.num_steps, self.num_controls))
        self.params_set = False

    def set_params(self, params):
        self.params = params
        self.params_set = True

    def solve(self):
        if not self.params_set:
            raise ValueError("MPPI parameters are not set. Cannot solve.")

        key = self.key
        params = self.params
        x0 = params['x0']
        u_std = params['u_std']
        lambda_weight = params['lambda_weight']
        num_opt = params['num_opt']

        for k in range(num_opt):
            # Fixing the shape of keys here to match self.num_control_rollouts
            keys = random.split(key, self.num_control_rollouts)

            # Generate noise with correct dimensions (self.num_control_rollouts, self.num_steps, self.num_controls)
            noise = random.normal(keys, (self.num_control_rollouts, self.num_steps, self.num_controls)) * u_std

            # Adjust shape of u_seq to ensure consistent operations
            u_seq_noisy = self.u_seq + noise

            # Run rollouts with the noisy control sequences
            costs = vmap_rollout(keys, x0, u_seq_noisy, params)

            # Compute weights
            beta = jnp.min(costs)
            exp_costs = jnp.exp(-1.0 / lambda_weight * (costs - beta))
            weights = exp_costs / jnp.sum(exp_costs)

            # Update control sequence
            weighted_noise = jnp.tensordot(weights, noise, axes=1)
            self.u_seq += weighted_noise

        return self.u_seq


    def shift_and_update(self, new_x0, u_seq, num_shifts=1):
        self.params['x0'] = new_x0.copy()
        self.u_seq = jnp.roll(u_seq, -num_shifts, axis=0)
        self.u_seq = self.u_seq.at[-num_shifts:].set(0.0)

# Main execution
if __name__ == "__main__":
    num_controls = 6
    num_states = 12
    cfg = Config(T=1.0, dt=0.02, num_control_rollouts=1024, num_controls=num_controls, num_states=num_states, seed=1)

    x0 = jnp.zeros(12)
    xgoal = jnp.array([2, -1, 3, 0, 0, 0, 0.1, -0.1, 0.3, 0, 0, 0])

    mppi_params = dict(
        # Task specification
        dt=cfg.dt,
        x0=x0,  # Start state
        xgoal=xgoal,  # Goal position

        # For risk-aware min time planning
        goal_tolerance=0.001,
        dist_weight=2000,  # Weight for dist-to-goal cost.
        lambda_weight=20.0,  # Temperature param in MPPI
        num_opt=5,  # Number of steps in each solve() function call.

        # Control and sample specification
        u_std=jnp.array([1.0, 1.0, 1.0, 0.03, 0.03, 0.03]) * 0.1,  # Noise std for sampling controls
    )

    mppi_controller = MPPI_JAX(cfg, key)
    mppi_controller.set_params(mppi_params)

    # Loop
    max_steps = 500
    xhist = np.zeros((max_steps + 1, num_states)) * np.nan
    uhist = np.zeros((max_steps, num_controls)) * np.nan
    xhist[0] = x0

    for t in range(max_steps):
        # Solve
        u_seq = mppi_controller.solve()
        u_curr = u_seq[0]
        phi, theta, psi = xhist[t, 6:9]
        gravity_vector_world = np.array([0, 0, 9.81 * 2.302499999999999])
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                       np.cos(phi)*np.cos(theta)]
        ])
        gravity_body = np.dot(R.T, gravity_vector_world)  # Rotate gravity to body frame
        u_curr = u_curr + gravity_body[:6]  # Adjust controls for gravity
        uhist[t] = u_curr

        # Simulate state forward
        xhist[t+1, :] = dynamics_update_sim(xhist[t, :], u_curr, cfg.dt)
        # Update MPPI state (x0, u_seq)
        mppi_controller.shift_and_update(xhist[t+1], u_seq, num_shifts=1)

    # Plotting code remains the same
    # Assuming xgoal is your goal position and it has appropriate values for each state
    x_goal, y_goal, z_goal = xgoal[:3]
    roll_goal, pitch_goal, yaw_goal = xgoal[6:9]

    fig, axs = plt.subplots(4, 3, figsize=(12, 9))  # Create subplots

    # Plot X with Goal
    axs[0][0].plot(xhist[:, 0], label='x')
    axs[0][0].axhline(x_goal, color='green', linestyle='--', label='X Goal')
    axs[0][0].set_title('X')
    axs[0][0].set_xlabel('Time Steps')
    axs[0][0].set_ylabel('m')
    axs[0][0].legend()

    # Plot Y with Goal
    axs[0][1].plot(xhist[:, 1], label='y')
    axs[0][1].axhline(y_goal, color='green', linestyle='--', label='Y Goal')
    axs[0][1].set_title('Y')
    axs[0][1].set_xlabel('Time Steps')
    axs[0][1].set_ylabel('m')
    axs[0][1].legend()

    # Plot Z with Goal
    axs[0][2].plot(xhist[:, 2], label='z')
    axs[0][2].axhline(z_goal, color='green', linestyle='--', label='Z Goal')
    axs[0][2].set_title('Z')
    axs[0][2].set_xlabel('Time Steps')
    axs[0][2].set_ylabel('m')
    axs[0][2].legend()

    # Plot Roll with Goal
    axs[1][0].plot(xhist[:, 6]*180/np.pi, label='roll')
    axs[1][0].axhline(roll_goal*180/np.pi, color='green', linestyle='--', label='Roll Goal')
    axs[1][0].set_title('Roll')
    axs[1][0].set_xlabel('Time Steps')
    axs[1][0].set_ylabel('Angle (degrees)')
    axs[1][0].legend()

    # Plot Pitch with Goal
    axs[1][1].plot(xhist[:, 7]*180/np.pi, label='pitch')
    axs[1][1].axhline(pitch_goal*180/np.pi, color='green', linestyle='--', label='Pitch Goal')
    axs[1][1].setTitle('Pitch')
    axs[1][1].set_xlabel('Time Steps')
    axs[1][1].set_ylabel('Angle (degrees)')
    axs[1][1].legend()

    # Plot Yaw with Goal
    axs[1][2].plot(xhist[:, 8]*180/np.pi, label='yaw')
    axs[1][2].axhline(yaw_goal*180/np.pi, color='green', linestyle='--', label='Yaw Goal')
    axs[1][2].set_title('Yaw')
    axs[1][2].set_xlabel('Time Steps')
    axs[1][2].set_ylabel('Angle (degrees)')
    axs[1][2].legend()

    # Plot Fx
    axs[2][0].plot(uhist[:, 0], label='Fx')
    axs[2][0].set_title('Control Fx')
    axs[2][0].set_xlabel('Time Steps')
    axs[2][0].set_ylabel('N')
    axs[2][0].legend()

    # Plot Fy
    axs[2][1].plot(uhist[:, 1], label='Fy')
    axs[2][1].set_title('Control Fy')
    axs[2][1].set_xlabel('Time Steps')
    axs[2][1].set_ylabel('N')
    axs[2][1].legend()

    # Plot Fz
    axs[2][2].plot(uhist[:, 2], label='Fz')
    axs[2][2].set_title('Control Fz')
    axs[2][2].set_xlabel('Time Steps')
    axs[2][2].set_ylabel('N')
    axs[2][2].legend()

    # Plot Mx
    axs[3][0].plot(uhist[:, 3], label='Mx')
    axs[3][0].set_title('Control Mx')
    axs[3][0].set_xlabel('Time Steps')
    axs[3][0].set_ylabel('Nm')
    axs[3][0].legend()

    # Plot My
    axs[3][1].plot(uhist[:, 4], label='My')
    axs[3][1].set_title('Control My')
    axs[3][1].set_xlabel('Time Steps')
    axs[3][1].set_ylabel('Nm')
    axs[3][1].legend()

    # Plot Mz
    axs[3][2].plot(uhist[:, 5], label='Mz')
    axs[3][2].set_title('Control Mz')
    axs[3][2].set_xlabel('Time Steps')
    axs[3][2].set_ylabel('Nm')
    axs[3][2].legend()

    plt.tight_layout()  # Adjusts the subplots to fit in the figure area
    plt.show()
