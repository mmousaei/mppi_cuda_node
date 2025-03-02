#!/usr/bin/env python
"""
Full MPPI implementation with 3D obstacle avoidance cost.
Obstacles are defined as spheres with a 3D center and a radius.
"""

import numpy as np
import math
import copy
import numba
import time
from numba import cuda, float32, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import matplotlib.pyplot as plt

import os
import sys

# Get the absolute path of the directory containing mpc
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "mpc"))
sys.path.append(parent_dir)

# Now import your module using absolute import
from acados.acados_mpc import MPC

# Information about your GPU
gpu = cuda.get_current_device()
max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
max_blocks = gpu.MAX_GRID_DIM_X
max_rec_blocks = rec_max_control_rollouts = int(1e6)  # Theoretical max number of control rollouts
rec_min_control_rollouts = 100

# Constant for contact-related dynamics (unused in obstacle avoidance)
CONTACT_NORMAL = np.array([-1, 0, 0], dtype=np.float32)

class Config:
    """ Fixed configurations used throughout execution. """
    
    def __init__(self, 
                 T=0.5,                # Horizon in seconds
                 dt=0.02,              # Time step in seconds
                 num_control_rollouts=1024,  # Number of control sequences to sample
                 num_controls=6,
                 num_states=12,
                 num_vis_state_rollouts=20,  # Number of visualization rollouts
                 seed=1):
        self.seed = seed
        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt)
        self.max_threads_per_block = max_threads_per_block
        self.num_controls = num_controls
        self.num_states = num_states

        assert T > 0
        assert dt > 0
        assert T > dt
        assert self.num_steps > 0

        # Limit the number of control rollouts to available GPU blocks
        self.num_control_rollouts = num_control_rollouts
        if self.num_control_rollouts > rec_max_control_rollouts:
            self.num_control_rollouts = rec_max_control_rollouts
            print("MPPI Config: Clipping num_control_rollouts to {}.".format(rec_max_control_rollouts))
        elif self.num_control_rollouts < rec_min_control_rollouts:
            self.num_control_rollouts = rec_min_control_rollouts
            print("MPPI Config: Clipping num_control_rollouts to minimum {}.".format(rec_min_control_rollouts))
        
        # For visualization, clip the number of state rollouts
        self.num_vis_state_rollouts = min(num_vis_state_rollouts, self.num_control_rollouts)
        self.num_vis_state_rollouts = max(1, self.num_vis_state_rollouts)
        print("num_steps: ", self.num_steps)

DEFAULT_OBS_COST = 1e3
DEFAULT_DIST_WEIGHT = 10

# Example stage and terminal cost weights (you may adjust these)
STAGE_COST_WEIGHTS = np.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0], dtype=np.float32)
TERMINAL_COST_WEIGHTS = np.array([1000, 1000, 2000, 0, 0, 0, 5000, 5000, 10000, 0, 0, 0], dtype=np.float32)

def dynamics_update_sim(x, u, dt):
    # Simple Newton-Euler dynamics for a hexarotor
    I_xx = 0.115125971
    I_yy = 0.116524229
    I_zz = 0.230387752
    mass = 7.00
    g = 9.81

    x_next = x.copy()
    x_next[0] += dt * x[3]
    x_next[1] += dt * x[4]
    x_next[2] += dt * x[5]
  
    x_next[3] += dt * ((1/mass) * u[0] - g * (np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) + np.sin(x[6]) * np.sin(x[8])))
    x_next[4] += dt * ((1/mass) * u[1] - g * (np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) - np.sin(x[6]) * np.cos(x[8])))
    x_next[5] += dt * ((1/mass) * u[2] - g * (np.cos(x[6]) * np.cos(x[7])))
  
    x_next[6] += dt * (x[9] + x[10]*(math.sin(x[6])*math.tan(x[7])) + x[11]*(math.cos(x[6])*math.tan(x[7])))
    x_next[7] += dt * (x[10]*math.cos(x[6]) - x[11]*math.sin(x[6]))
    x_next[8] += dt * (x[10]*math.sin(x[6])/math.cos(x[7]) + x[11]*math.cos(x[6])/math.cos(x[7]))
  
    x_next[9]  += dt * ((1/I_xx) * (u[3] + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
    x_next[10] += dt * ((1/I_yy) * (u[4] - I_xx * x[9] * x[11] + I_zz * x[9] * x[11]))
    x_next[11] += dt * ((1/I_zz) * (u[5] + I_xx * x[9] * x[10] - I_yy * x[9] * x[10]))
    return x_next

# Device functions for cost computation
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def stage_cost(dist2, dist_weight):
    return dist_weight * dist2

@cuda.jit('float32(float32, boolean)', device=True, inline=True)
def term_cost(dist2, goal_reached):
    return (1 - np.float32(goal_reached)) * dist2

@cuda.jit(device=True, fastmath=True)
def dynamics_update(x, u, dt, contact_normal, inertia_mass):
    # Simplified dynamics update ignoring contact forces for this example
    contact_normal_sq = 1
    A = -1
    B = 0
    C = 0
    D = 15
    ABC_sq = 1

    I_xx = inertia_mass[0]
    I_yy = inertia_mass[1]
    I_zz = inertia_mass[2]
    mass = inertia_mass[3]

    # No contact forces in this simplified model
    c = -300
    g = 9.81

    fx_total = u[0]
    fy_total = u[1]
    fz_total = u[2]
    mx_total = u[3]
    my_total = u[4]
    mz_total = u[5]

    sin_phi = math.sin(x[6])
    cos_phi = math.cos(x[6])
    sin_theta = math.sin(x[7])
    cos_theta = math.cos(x[7])
    sin_psi = math.sin(x[8])
    cos_psi = math.cos(x[8])

    x[0] += dt * x[3]
    x[1] += dt * x[4]
    x[2] += dt * x[5]

    x[3] += dt * ((1/mass) * fx_total - g * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi))
    x[4] += dt * ((1/mass) * fy_total - g * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi))
    x[5] += dt * ((1/mass) * fz_total - g * cos_phi * cos_theta)

    x[6] += dt * (x[9] + x[10]*(math.sin(x[6])*math.tan(x[7])) + x[11]*(math.cos(x[6])*math.tan(x[7])))
    x[7] += dt * (x[10]*math.cos(x[6]) - x[11]*math.sin(x[6]))
    x[8] += dt * (x[10]*math.sin(x[6])/math.cos(x[7]) + x[11]*math.cos(x[6])/math.cos(x[7]))
  
    x[9]  += dt * ((1/I_xx) * (mx_total + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
    x[10] += dt * ((1/I_yy) * (my_total - I_xx * x[9] * x[11] + I_zz * x[9] * x[11]))
    x[11] += dt * ((1/I_zz) * (mz_total + I_xx * x[9] * x[10] - I_yy * x[9] * x[10]))

# MPPI controller implemented using Numba on the GPU
class MPPI_Numba(object):
    """
    MPPI implementation using information theoretic control.
    """
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

        self.max_threads_per_block = cfg.max_threads_per_block

        # Initialize device variables
        self.noise_samples_d = None
        self.u_cur_d = None
        self.u_prev_d = None
        self.costs_d = None
        self.weights_d = None
        self.rng_states_d = None
        self.state_rollout_batch_d = None

        self.last_noise_d = None
        self.use_ou = False
        self.theta = 2
        self.mu = 0.0
        self.sigma = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.03]) * 0.2
        self.delta_t = self.cfg.dt
        self.ou_alpha = 0.7
        self.ou_scale = 1
        self.d_ou_scale = 0.5
        self.sys_noise = np.array([0.1, 0.1, 0.1, 0.001, 0.001, 0.001])
        self.dz = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
        self.umin = np.array([-20, -20, -40, -0.1, -0.1, -0.1])
        self.umax = np.array([20, 20, 40, 0.1, 0.1, 0.1])
        self.last_controls = np.zeros((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
        self.last_controls_d = cuda.to_device(self.last_controls.astype(np.float32))
        
        self.contact_normal = np.array([-1, 0, 0])
        self.contact_point = np.array([15, 0, 0])
        self.contact_normal_sq = self.contact_normal[0]**2 + self.contact_normal[1]**2 + self.contact_normal[2]**2
        self.A = self.contact_normal[0]
        self.B = self.contact_normal[1]
        self.C = self.contact_normal[2]
        self.D = -self.A * self.contact_point[0] - self.B * self.contact_point[1] - self.C * self.contact_point[2]
        self.ABC_sq = math.sqrt(self.A**2 + self.B**2 + self.C**2)
        
        self.device_var_initialized = False
        self.reset()

    def reset(self):
        self.u_seq0 = np.zeros((self.num_steps, self.num_controls), dtype=np.float32)
        mass = 7.00
        g = 9.81
        self.u_seq0[:, 2] = mass * g  # Hover thrust in z-direction
        self.params = None
        self.params_set = False
        self.u_prev_d = None
        self.last_noise_d = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
        self.init_device_vars_before_solving()

    def init_device_vars_before_solving(self):
        if not self.device_var_initialized:
            t0 = time.time()
            self.noise_samples_d = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
            self.u_cur_d = cuda.to_device(self.u_seq0)
            self.u_prev_d = cuda.to_device(self.u_seq0)
            self.costs_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
            self.weights_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
            self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts * self.num_steps, seed=self.seed)
            self.state_rollout_batch_d = cuda.device_array((self.num_vis_state_rollouts, self.num_steps + 1, self.num_states), dtype=np.float32)
            self.device_var_initialized = True
            print("MPPI planner has initialized GPU memory after {} s".format(time.time() - t0))

    def set_params(self, params):
        self.params = copy.deepcopy(params)
        self.params_set = True

    def check_solve_conditions(self):
        if not self.params_set:
            print("MPPI parameters are not set. Cannot solve")
            return False
        if not self.device_var_initialized:
            print("Device variables not initialized. Cannot solve.")
            return False
        return True

    def solve(self):
        if not self.check_solve_conditions():
            print("MPPI solve condition not met. Cannot solve. Return")
            return
        return self.solve_with_nominal_dynamics()

    def change_goal(self, goal):
        self.params["xgoal"] = goal

    def move_mppi_task_vars_to_device(self):
        vrange_d = cuda.to_device(self.params['vrange'].astype(np.float32))
        wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
        xgoal_d = cuda.to_device(self.params['xgoal'].astype(np.float32))
        goal_tolerance_d = np.float32(self.params['goal_tolerance'])
        lambda_weight_d = np.float32(self.params['lambda_weight'])
        u_std_d = cuda.to_device(self.params['u_std'].astype(np.float32))
        x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
        dt_d = np.float32(self.params['dt'])
        cost_weights_d = cuda.to_device(self.params['weights'].astype(np.float32))
        inertia_mass_d = cuda.to_device(self.params['inertia_mass'].astype(np.float32))
        
        # For 3D obstacles, obs_pos_d must be an (N x 3) array.
        if "obstacle_positions" in self.params:
            obs_pos_d = cuda.to_device(self.params['obstacle_positions'].astype(np.float32))
        else:
            obs_pos_d = np.array([[1e5, 1e5, 1e5]], dtype=np.float32)
        if "obstacle_radius" in self.params:
            obs_r_d = cuda.to_device(self.params['obstacle_radius'].astype(np.float32))
        else:
            obs_r_d = np.array([0], dtype=np.float32)
        
        obs_cost_d = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params 
                                  else self.params['obs_penalty'])
        return (vrange_d, wrange_d, xgoal_d, goal_tolerance_d, lambda_weight_d,
                u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d, cost_weights_d, inertia_mass_d)

    def solve_with_nominal_dynamics(self):
        (vrange_d, wrange_d, xgoal_d, goal_tolerance_d, lambda_weight_d,
         u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d, cost_weights_d, inertia_mass_d) = self.move_mppi_task_vars_to_device()
        
        dist_to_goal_d = cuda.device_array(6, dtype=np.float32)
        coef_dist_to_goal = np.array([1, 1, 5, 0.03, 0.03, 0.03], dtype=np.float32) * 0.1
        dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']
        
        for k in range(self.params['num_opt']):
            if self.use_ou:
                dist_to_goal = (self.params['xgoal'][:6] - self.params['x0'][:6])**2
                self.sample_noise_ou_numba[self.num_control_rollouts, self.num_steps](
                    self.rng_states_d, self.theta, self.mu, self.sigma, self.dt, self.noise_samples_d)
            else:
                dist_to_goal = np.abs(self.params['xgoal'][:6] - self.params['x0'][:6])
                u_std_scaled = np.minimum(u_std_d, coef_dist_to_goal * dist_to_goal)
                self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
                    self.rng_states_d, u_std_d, self.noise_samples_d)
            
            self.rollout_numba[self.num_control_rollouts, 1](
                inertia_mass_d, vrange_d, wrange_d, xgoal_d, obs_cost_d, obs_pos_d, obs_r_d,
                goal_tolerance_d, lambda_weight_d, u_std_d, x0_d, dt_d, dist_weight,
                cost_weights_d, self.noise_samples_d, self.u_cur_d, self.costs_d)
            self.u_prev_d = self.u_cur_d
            self.update_useq_numba[1, 32](lambda_weight_d, self.costs_d, self.noise_samples_d, self.weights_d, vrange_d, wrange_d, self.u_cur_d)
            cost = self.costs_d.copy_to_host()
        return self.u_cur_d.copy_to_host()

    def shift_and_update(self, new_x0, u_cur, num_shifts=1):
        self.params["x0"] = new_x0.copy()
        self.shift_optimal_control_sequence(u_cur, num_shifts)
        self.last_controls = u_cur
        self.last_controls_d = cuda.to_device(self.last_controls.astype(np.float32))

    def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
        u_cur_shifted = u_cur.copy()
        u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
        self.u_cur_d = cuda.to_device(u_cur_shifted.astype(np.float32))

    def get_state_rollout(self):
        assert self.params_set, "MPPI parameters are not set"
        if not self.device_var_initialized:
            print("Device variables not initialized. Cannot run mppi.")
            return
        vrange_d = cuda.to_device(self.params['vrange'].astype(np.float32))
        wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
        x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
        dt_d = np.float32(self.params['dt'])
        self.get_state_rollout_across_control_noise[self.num_vis_state_rollouts, 1](
            self.state_rollout_batch_d, x0_d, dt_d, self.noise_samples_d, vrange_d, wrange_d, self.u_prev_d, self.u_cur_d)
        return self.state_rollout_batch_d.copy_to_host()

    # --- GPU Kernels ---
    @staticmethod
    @cuda.jit(fastmath=True)
    def rollout_numba(inertia_mass_d, vrange_d, wrange_d, xgoal_d, obs_cost_d, obs_pos_d, obs_r_d,
                      goal_tolerance_d, lambda_weight_d, u_std_d, x0_d, dt_d,
                      dist_weight_d, cost_weights_d, noise_samples_d, u_cur_d, costs_d):
        """
        Each block simulates one control rollout.
        """
        bid = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        costs_d[bid] = 0.0

        x_curr = cuda.local.array(12, numba.float32)
        for i in range(12):
            x_curr[i] = x0_d[i]
        timesteps = len(u_cur_d)
        goal_reached = False
        goal_tolerance_d2 = goal_tolerance_d * goal_tolerance_d
        dist_to_goal2 = 1e9
        u_nom = cuda.local.array(6, numba.float32)

        u_prev = cuda.local.array(6, numba.float32)
        for i in range(6):
            u_prev[i] = u_cur_d[0, i]

        for t in range(timesteps):
            u_nom[0] = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
            u_nom[1] = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
            u_nom[2] = u_cur_d[t, 2] + noise_samples_d[bid, t, 2]
            u_nom[3] = u_cur_d[t, 3] + noise_samples_d[bid, t, 3]
            u_nom[4] = u_cur_d[t, 4] + noise_samples_d[bid, t, 4]
            u_nom[5] = u_cur_d[t, 5] + noise_samples_d[bid, t, 5]
            u_noisy = u_nom

            dynamics_update(x_curr, u_noisy, dt_d, CONTACT_NORMAL, inertia_mass_d)

            dist_to_goal2 = (cost_weights_d[0]*((xgoal_d[0]-x_curr[0])**2) +
                             cost_weights_d[1]*((xgoal_d[1]-x_curr[1])**2) +
                             cost_weights_d[2]*((xgoal_d[2]-x_curr[2])**2) +
                             cost_weights_d[3]*((xgoal_d[3]-x_curr[3])**2) +
                             cost_weights_d[4]*((xgoal_d[4]-x_curr[4])**2) +
                             cost_weights_d[5]*((xgoal_d[5]-x_curr[5])**2) +
                             cost_weights_d[6]*((xgoal_d[6]-x_curr[6])**2) +
                             cost_weights_d[7]*((xgoal_d[7]-x_curr[7])**2) +
                             cost_weights_d[8]*((xgoal_d[8]-x_curr[8])**2) +
                             cost_weights_d[9]*((xgoal_d[9]-x_curr[9])**2) +
                             cost_weights_d[10]*((xgoal_d[10]-x_curr[10])**2) +
                             cost_weights_d[11]*((xgoal_d[11]-x_curr[11])**2) +
                             cost_weights_d[12]*((u_nom[0]**2) + (u_nom[1]**2) +
                                                 ((u_nom[2] - inertia_mass_d[3]*9.81)**2)) +
                             cost_weights_d[13]*((u_nom[3]**2) + (u_nom[4]**2) + (u_nom[5]**2)))
            costs_d[bid] += stage_cost(dist_to_goal2, dist_weight_d)

            # --- 3D Obstacle Avoidance Cost ---
            num_obs = int(obs_r_d.size)
            for obs_i in range(num_obs):
                ox = obs_pos_d[obs_i, 0]
                oy = obs_pos_d[obs_i, 1]
                oz = obs_pos_d[obs_i, 2]
                dx = x_curr[0] - ox
                dy = x_curr[1] - oy
                dz = x_curr[2] - oz
                dist_sq = dx * dx + dy * dy + dz * dz
                r_sq = obs_r_d[obs_i] * obs_r_d[obs_i]
                if dist_sq < r_sq:
                    costs_d[bid] += (obs_cost_d * obs_cost_d) 
                # else:
                #     costs_d[bid] += obs_cost_d * math.exp(-dist_sq / r_sq)

            if dist_to_goal2 <= goal_tolerance_d2:
                goal_reached = True
                break

        costs_d[bid] += cost_weights_d[16] * term_cost(dist_to_goal2, goal_reached)
        for t in range(timesteps):
            costs_d[bid] += cost_weights_d[14] * lambda_weight_d * (
                (u_cur_d[t,0]/(u_std_d[0]**2)) * noise_samples_d[bid, t,0] +
                (u_cur_d[t,1]/(u_std_d[1]**2)) * noise_samples_d[bid, t,1] +
                (u_cur_d[t,2]/(u_std_d[2]**2)) * noise_samples_d[bid, t,2] +
                cost_weights_d[15]*(
                    (u_cur_d[t,3]/(u_std_d[3]**2)) * noise_samples_d[bid, t,3] +
                    (u_cur_d[t,4]/(u_std_d[4]**2)) * noise_samples_d[bid, t,4] +
                    (u_cur_d[t,5]/(u_std_d[5]**2)) * noise_samples_d[bid, t,5]
                )
            )

    @staticmethod
    @cuda.jit(fastmath=True)
    def update_useq_numba(lambda_weight_d, costs_d, noise_samples_d, weights_d, vrange_d, wrange_d, u_cur_d):
        tid = cuda.threadIdx.x
        num_threads = cuda.blockDim.x
        numel = len(noise_samples_d)
        gap = int(math.ceil(numel / num_threads))
        starti = min(tid * gap, numel)
        endi = min(starti + gap, numel)
        if starti < numel:
            weights_d[starti] = costs_d[starti]
        for i in range(starti, endi):
            weights_d[starti] = min(weights_d[starti], costs_d[i])
        cuda.syncthreads()

        s = gap
        while s < numel:
            if (starti % (2 * s) == 0) and ((starti + s) < numel):
                weights_d[starti] = min(weights_d[starti], weights_d[starti + s])
            s *= 2
            cuda.syncthreads()

        beta = weights_d[0]
        for i in range(starti, endi):
            weights_d[i] = math.exp(-1. / lambda_weight_d * (costs_d[i] - beta))
        cuda.syncthreads()

        for i in range(starti, endi):
            costs_d[i] = weights_d[i]
        cuda.syncthreads()
        for i in range(starti + 1, endi):
            costs_d[starti] += costs_d[i]
        cuda.syncthreads()
        s = gap
        while s < numel:
            if (starti % (2 * s) == 0) and ((starti + s) < numel):
                costs_d[starti] += costs_d[starti + s]
            s *= 2
            cuda.syncthreads()

        for i in range(starti, endi):
            weights_d[i] /= costs_d[0]
        cuda.syncthreads()

        timesteps = len(u_cur_d)
        for t in range(timesteps):
            for i in range(starti, endi):
                cuda.atomic.add(u_cur_d, (t, 0), weights_d[i] * noise_samples_d[i, t, 0])
                cuda.atomic.add(u_cur_d, (t, 1), weights_d[i] * noise_samples_d[i, t, 1])
                cuda.atomic.add(u_cur_d, (t, 2), weights_d[i] * noise_samples_d[i, t, 2])
                cuda.atomic.add(u_cur_d, (t, 3), weights_d[i] * noise_samples_d[i, t, 3])
                cuda.atomic.add(u_cur_d, (t, 4), weights_d[i] * noise_samples_d[i, t, 4])
                cuda.atomic.add(u_cur_d, (t, 5), weights_d[i] * noise_samples_d[i, t, 5])
        cuda.syncthreads()

    @staticmethod
    @cuda.jit(fastmath=True)
    def get_state_rollout_across_control_noise(state_rollout_batch_d, x0_d, dt_d, noise_samples_d, vrange_d, wrange_d, u_prev_d, u_cur_d):
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        timesteps = len(u_cur_d)
        if bid == 0:
            x_curr = cuda.local.array(3, numba.float32)
            for i in range(3):
                x_curr[i] = x0_d[i]
                state_rollout_batch_d[bid, 0, i] = x0_d[i]
            for t in range(timesteps):
                u_nom = u_cur_d[t, :]
                dynamics_update(x_curr, u_nom, dt_d, CONTACT_NORMAL)
                state_rollout_batch_d[bid, t+1, 0] = x_curr[0]
                state_rollout_batch_d[bid, t+1, 1] = x_curr[1]
                state_rollout_batch_d[bid, t+1, 2] = x_curr[2]
        else:
            x_curr = cuda.local.array(3, numba.float32)
            for i in range(3):
                x_curr[i] = x0_d[i]
                state_rollout_batch_d[bid, 0, i] = x0_d[i]
            for t in range(timesteps):
                u_nom = cuda.local.array(6, numba.float32)
                u_nom[0] = u_prev_d[t, 0] + noise_samples_d[bid, t, 0]
                u_nom[1] = u_prev_d[t, 1] + noise_samples_d[bid, t, 1]
                u_nom[2] = u_prev_d[t, 2] + noise_samples_d[bid, t, 2]
                u_nom[3] = u_prev_d[t, 3] + noise_samples_d[bid, t, 3]
                u_nom[4] = u_prev_d[t, 4] + noise_samples_d[bid, t, 4]
                u_nom[5] = u_prev_d[t, 5] + noise_samples_d[bid, t, 5]
                u_noisy = u_nom
                dynamics_update(x_curr, u_noisy, dt_d, CONTACT_NORMAL)
                state_rollout_batch_d[bid, t+1, 0] = x_curr[0]
                state_rollout_batch_d[bid, t+1, 1] = x_curr[1]
                state_rollout_batch_d[bid, t+1, 2] = x_curr[2]

    @staticmethod
    @cuda.jit(fastmath=True)
    def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
        block_id = cuda.blockIdx.x
        thread_id = cuda.threadIdx.x
        abs_thread_id = cuda.grid(1)
        num_timesteps = noise_samples_d.shape[1]
        num_controls = noise_samples_d.shape[2]
        denom = 20
        for t in range(num_timesteps):
            for i in range(num_controls):
                scale = 1.0
                scaled_std = u_std_d[i] * scale
                noise_samples_d[block_id, t, i] = scaled_std * xoroshiro128p_normal_float32(rng_states, abs_thread_id)

    @staticmethod
    @cuda.jit(fastmath=True)
    def sample_noise_ou_numba(rng_states, theta, mu, sigma, dt, noise_samples_d):
        bid = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        num_controls = noise_samples_d.shape[2]
        abs_tid = bid * noise_samples_d.shape[1] + tid
        for i in range(num_controls):
            if tid == 0:
                prev_noise = mu
            else:
                prev_noise = noise_samples_d[bid, tid - 1, i]
            dx = theta * (mu - prev_noise) * dt + sigma[i] * math.sqrt(dt) * xoroshiro128p_normal_float32(rng_states, abs_tid)
            noise_samples_d[bid, tid, i] = prev_noise + dx

if __name__ == "__main__":
    num_controls = 6
    num_states = 12
    cfg = Config(
        T=2,                # Horizon length (s)
        dt=0.3,             # Time step (s)
        num_control_rollouts=1024 * 4,
        num_controls=6,
        num_states=12,
        num_vis_state_rollouts=1,
        seed=1
    )
    x0 = np.array([0, 0, 0, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    xgoal = np.array([2, 2, 1, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    
    # Define two 3D spherical obstacles.
    # For example, one centered at [1, 1, 0.5] with radius 0.5 and another at [2, -1, 0.5] with radius 0.75.
    obs_positions = np.array([[1.0, 1.0, 0.5],
                              [2.0, -1.0, 0.5]], dtype=np.float32)
    obs_radius = np.array([0.5, 0.5], dtype=np.float32)
    
    mppi_params = {
        'dt': cfg.dt,
        'x0': x0,
        'xgoal': xgoal,
        'goal_tolerance': 0.001,
        'dist_weight': 2000,
        'lambda_weight': 10,
        'num_opt': 8,
        'u_std': np.array([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]),
        'vrange': np.array([-10.0, 10.0]),
        'wrange': np.array([-0.1, 0.1]),
        'weights': np.array([
            9550, 9550, 24840,
            10, 10, 10,
            25500, 25500, 25500,
            1, 1, 1,
            1, 100, 1, 100, 9000
        ]),
        "inertia_mass": np.array([0.21, 0.21, 0.4, 6.15]),
        "obstacle_positions": obs_positions,
        "obstacle_radius": obs_radius,
        "obs_penalty": 1e10
    }

    mppi_controller = MPPI_Numba(cfg)
    mppi_controller.set_params(mppi_params)

    use_mpc = False
    max_steps = 500

    mpc_params = {
        'inertia': np.array([0.115125971, 0.116524229, 0.230387752]),
        'mass': 7.00,
        'horizon': 30,
        'gravity': 9.81,
        'max_force': 10.0,
        'max_torque': 1,
        'control_weight': 0.4,
        'tracking_weight_pos': 50,
        'tracking_weight_vel': 3,
        'tracking_weight_att': 30,
        'tracking_weight_ang_vel': 5,
        'terminal_weight': 1,
        'smoothness_weight': 0.05,
        'dt': 0.01
    }
    
    from acados.acados_mpc import MPC
    mpc = MPC(mpc_params)

    def hex_dynamics(x, u, mppi_params):
        p, v, Psi, omega = np.split(x, 4)
        f_T, m_T = u[:3], u[3:]
        phi, theta, psi = Psi
        J = np.diag(mppi_params['inertia_mass'][:3])
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
             np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),
             np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi),
             np.cos(phi)*np.cos(theta)]
        ])
        gravity_world = np.array([0, 0, -9.81])
        gravity_body = np.dot(R.T, gravity_world)
        nu = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        p_dot = v
        v_dot = (1/mppi_params['inertia_mass'][3]) * f_T + gravity_body
        psi_dot = np.dot(nu, omega)
        omega_dot = np.dot(np.linalg.inv(J), m_T - np.cross(omega, np.dot(J, omega)))
        return np.concatenate([p_dot, v_dot, psi_dot, omega_dot])
    
    def forward_simulate_for_mpc_target(optimal_control_seq, current_state, mppi_params):
        mppi_u = optimal_control_seq[0, :].copy()
        next_state = dynamics_update_rk4(current_state.copy(), mppi_u, mppi_params['dt'], mppi_params)
        return next_state

    def dynamics_update_rk4(state, control_inputs, dt, mppi_params):
        k1 = hex_dynamics(state, control_inputs, mppi_params) * dt
        k2 = hex_dynamics(state + k1/2, control_inputs, mppi_params) * dt 
        k3 = hex_dynamics(state + k2/2, control_inputs, mppi_params) * dt
        k4 = hex_dynamics(state + k3, control_inputs, mppi_params) * dt
        next_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        return next_state

    xhist = np.zeros((max_steps+1, num_states)) * np.nan
    uhist = np.zeros((max_steps, num_controls)) * np.nan
    mpctargethist = np.zeros((max_steps+1, num_states)) * np.nan
    xhist[0] = x0
    mpctargethist[0] = x0

    for t in range(max_steps):
        useq = mppi_controller.solve()
        u_curr = useq[0]
        phi, theta, psi = xhist[t, 6:9]
        gravity_vector_world = np.array([0, 0, 9.81 * 7.00])
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                       np.cos(phi)*np.cos(theta)]
        ])
        gravity_body = np.dot(R.T, gravity_vector_world)
        if t % 10 == 0:
            mpc_target = forward_simulate_for_mpc_target(useq, xhist[t, :], mppi_params)
        u_mpc = mpc.compute_control(xhist[t, :], mpc_target, np.zeros(6), mpc_params['dt'])
        mpctargethist[t+1, :] = mpc_target.copy()
        if use_mpc:
            uhist[t] = u_mpc
        else:
            uhist[t] = u_curr.copy()
        if use_mpc:
            xhist[t+1, :] = dynamics_update_rk4(xhist[t, :], u_mpc, mpc_params['dt'], mppi_params)
        else:
            xhist[t+1, :] = dynamics_update_rk4(xhist[t, :], u_curr, cfg.dt, mppi_params)
        print(t)
        mppi_controller.shift_and_update(xhist[t+1], useq, num_shifts=1)

    # Plot results
    fig, axs = plt.subplots(6, 3, figsize=(12, 9))
    axs[0][0].plot(xhist[:, 0], label='x')
    axs[0][0].axhline(xgoal[0], color='green', linestyle='--', label='X Goal')
    axs[0][0].set_title('X')
    axs[0][0].set_xlabel('Time Steps')
    axs[0][0].set_ylabel('m')
    axs[0][0].legend()

    axs[0][1].plot(xhist[:, 1], label='y')
    axs[0][1].axhline(xgoal[1], color='green', linestyle='--', label='Y Goal')
    axs[0][1].set_title('Y')
    axs[0][1].set_xlabel('Time Steps')
    axs[0][1].set_ylabel('m')
    axs[0][1].legend()

    axs[0][2].plot(xhist[:, 2], label='z')
    axs[0][2].axhline(xgoal[2], color='green', linestyle='--', label='Z Goal')
    axs[0][2].set_title('Z')
    axs[0][2].set_xlabel('Time Steps')
    axs[0][2].set_ylabel('m')
    axs[0][2].legend()

    axs[1][0].plot(xhist[:, 6]*180/np.pi, label='roll')
    axs[1][0].axhline(xgoal[6]*180/np.pi, color='green', linestyle='--', label='Roll Goal')
    axs[1][0].set_title('Roll')
    axs[1][0].set_xlabel('Time Steps')
    axs[1][0].set_ylabel('Angle (degrees)')
    axs[1][0].legend()

    axs[1][1].plot(xhist[:, 7]*180/np.pi, label='pitch')
    axs[1][1].axhline(xgoal[7]*180/np.pi, color='green', linestyle='--', label='Pitch Goal')
    axs[1][1].set_title('Pitch')
    axs[1][1].set_xlabel('Time Steps')
    axs[1][1].set_ylabel('Angle (degrees)')
    axs[1][1].legend()

    axs[1][2].plot(xhist[:, 8]*180/np.pi, label='yaw')
    axs[1][2].axhline(xgoal[8]*180/np.pi, color='green', linestyle='--', label='Yaw Goal')
    axs[1][2].set_title('Yaw')
    axs[1][2].set_xlabel('Time Steps')
    axs[1][2].set_ylabel('Angle (degrees)')
    axs[1][2].legend()

    axs[2][0].plot(uhist[:, 0], label='Fx')
    axs[2][0].set_title('Control Fx')
    axs[2][0].set_xlabel('Time Steps')
    axs[2][0].set_ylabel('N')
    axs[2][0].legend()

    axs[2][1].plot(uhist[:, 1], label='Fy')
    axs[2][1].set_title('Control Fy')
    axs[2][1].set_xlabel('Time Steps')
    axs[2][1].set_ylabel('N')
    axs[2][1].legend()

    axs[2][2].plot(uhist[:, 2], label='Fz')
    axs[2][2].set_title('Control Fz')
    axs[2][2].set_xlabel('Time Steps')
    axs[2][2].set_ylabel('N')
    axs[2][2].legend()

    axs[3][0].plot(uhist[:, 3], label='Mx')
    axs[3][0].set_title('Control Mx')
    axs[3][0].set_xlabel('Time Steps')
    axs[3][0].set_ylabel('Nm')
    axs[3][0].legend()

    axs[3][1].plot(uhist[:, 4], label='My')
    axs[3][1].set_title('Control My')
    axs[3][1].set_xlabel('Time Steps')
    axs[3][1].set_ylabel('Nm')
    axs[3][1].legend()

    axs[3][2].plot(uhist[:, 5], label='Mz')
    axs[3][2].set_title('Control Mz')
    axs[3][2].set_xlabel('Time Steps')
    axs[3][2].set_ylabel('Nm')
    axs[3][2].legend()

    axs[4][0].plot(mpctargethist[:, 0], label='X')
    axs[4][0].set_title('MPC Target X')
    axs[4][0].set_xlabel('Time Steps')
    axs[4][0].set_ylabel('m')
    axs[4][0].legend()

    axs[4][1].plot(mpctargethist[:, 1], label='Y')
    axs[4][1].set_title('MPC Target Y')
    axs[4][1].set_xlabel('Time Steps')
    axs[4][1].set_ylabel('m')
    axs[4][1].legend()

    axs[4][2].plot(mpctargethist[:, 2], label='Z')
    axs[4][2].set_title('MPC Target Z')
    axs[4][2].set_xlabel('Time Steps')
    axs[4][2].set_ylabel('m')
    axs[4][2].legend()

    axs[5][0].plot(mpctargethist[:, 6]*180/np.pi, label='roll')
    axs[5][0].set_title('MPC Target Roll')
    axs[5][0].set_xlabel('Time Steps')
    axs[5][0].set_ylabel('degrees')
    axs[5][0].legend()

    axs[5][1].plot(mpctargethist[:, 7]*180/np.pi, label='pitch')
    axs[5][1].set_title('MPC Target Pitch')
    axs[5][1].set_xlabel('Time Steps')
    axs[5][1].set_ylabel('degrees')
    axs[5][1].legend()

    axs[5][2].plot(mpctargethist[:, 8]*180/np.pi, label='yaw')
    axs[5][2].set_title('MPC Target Yaw')
    axs[5][2].set_xlabel('Time Steps')
    axs[5][2].set_ylabel('degrees')
    axs[5][2].legend()

    from mpl_toolkits.mplot3d import Axes3D  # ensure this import is at the top if not already present

    # Create a new figure for the 3D plot
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')

    # Plot the 3D trajectory from the state history (xhist)
    ax3d.plot(xhist[:, 0], xhist[:, 1], xhist[:, 2], label='Trajectory', marker='o', markersize=2, linewidth=1)

    # Plot the goal position
    ax3d.scatter(xgoal[0], xgoal[1], xgoal[2], color='green', marker='*', s=200, label='Goal')

    # Plot each obstacle as a sphere
    # u and v parameterize the sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    for i in range(len(obs_positions)):
        cx, cy, cz = obs_positions[i]
        r = obs_radius[i]
        xs = cx + r * np.outer(np.cos(u), np.sin(v))
        ys = cy + r * np.outer(np.sin(u), np.sin(v))
        zs = cz + r * np.outer(np.ones_like(u), np.cos(v))
        ax3d.plot_surface(xs, ys, zs, color='red', alpha=0.5, linewidth=0)

    # Set labels and title
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title("3D Trajectory and Obstacles")
    ax3d.legend()


    plt.tight_layout()
    plt.show()
