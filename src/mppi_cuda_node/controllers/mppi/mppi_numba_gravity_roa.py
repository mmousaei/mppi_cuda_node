#!/usr/bin/env python3

import numpy as np
import math
import copy
import time
import os
import sys
import matplotlib.pyplot as plt

import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

###############################################################################
# 1) A Stubbed "MPC" Class for Low-Level Control
###############################################################################
class MPC:
    """
    In a real setup, replace this with your 'acados_mpc.py' or your actual
    low-level short-horizon MPC code. Here we just do a trivial "hover control."
    """
    def __init__(self, params):
        self.params = params

    def compute_control(self, state, target_state, initial_guess, dt):
        """
        Return a 6D action that tries to hold altitude or so.
        Real code would call ACADOS solver, etc.
        """
        # For demonstration, let’s do a 'hover control' that tries to 
        # keep z at the target_state[2].
        # This is obviously not a real MPC. 
        mass = self.params.get("mass", 7.0)
        g = self.params.get("gravity", 9.81)
        # naive approach: if current z < target z, apply slightly more thrust, else less.
        z_err = target_state[2] - state[2]
        Fz_hover = mass*g + 10*z_err  # crude P-gain

        # Other controls zero
        control = np.array([0.0, 0.0, Fz_hover, 0.0, 0.0, 0.0], dtype=float)
        return control


###############################################################################
# 2) Basic Config Object for MPPI
###############################################################################
class Config:
    """
    Holds MPPI time horizon, dt, number of rollouts, etc.
    """
    def __init__(self, 
                 T=2.0,
                 dt=0.3,
                 num_control_rollouts=1024,
                 num_controls=6,
                 num_states=12,
                 num_vis_state_rollouts=1,
                 seed=1):

        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt)
        self.num_control_rollouts = num_control_rollouts
        self.num_controls = num_controls
        self.num_states = num_states
        self.num_vis_state_rollouts = num_vis_state_rollouts
        self.seed = seed

        print("num_steps:", self.num_steps)


###############################################################################
# 3) A Simple Hexarotor Dynamical Update (device function)
###############################################################################
@cuda.jit(device=True)
def hex_dynamics_inplace(x, u, dt, mass):
    """
    In-place update of 12D hexarotor state x with 6D control u over dt.
    Minimal version for demonstration; no orientation matrix, etc.
    state x = [px,py,pz, vx,vy,vz, roll, pitch, yaw, p,q,r]
    control u= [Fx,Fy,Fz,  Mx,My,Mz]
    """
    g = 9.81
    # Inertias (some constants)
    Ixx, Iyy, Izz = 0.115125971, 0.116524229, 0.230387752

    # position
    x[0] += dt*x[3]
    x[1] += dt*x[4]
    x[2] += dt*x[5]

    # velocity
    Fx, Fy, Fz = u[0], u[1], u[2]
    roll, pitch, yaw = x[6], x[7], x[8]

    # naive gravity projection
    # realistic approach would do cos(roll)*cos(pitch), etc.
    x[3] += dt*((1.0/mass)*Fx - g*(0.0))  # ignoring tilt
    x[4] += dt*((1.0/mass)*Fy - g*(0.0))
    x[5] += dt*((1.0/mass)*Fz - g*(1.0))

    # orientation
    x[6] += dt*x[9]
    x[7] += dt*x[10]
    x[8] += dt*x[11]

    # angular rates
    Mx, My, Mz = u[3], u[4], u[5]
    p_,q_,r_ = x[9], x[10], x[11]
    x[9] += dt*((1.0/Ixx)*(Mx + (Iyy - Izz)*q_*r_))
    x[10]+= dt*((1.0/Iyy)*(My + (Izz - Ixx)*p_*r_))
    x[11]+= dt*((1.0/Izz)*(Mz + (Ixx - Iyy)*p_*q_))


###############################################################################
# 4) Numba Device Functions for Stage / Terminal Cost
###############################################################################
@cuda.jit(device=True, inline=True)
def l2norm_sq(x0, x1, x2):
    return x0*x0 + x1*x1 + x2*x2

@cuda.jit(device=True, inline=True)
def stage_cost(dist2, dist_weight):
    """
    Basic stage cost = dist_weight * dist2
    """
    return dist_weight*dist2

@cuda.jit(device=True, inline=True)
def term_cost(dist2, goal_reached):
    """
    Terminal cost = zero if goal_reached, else dist2
    """
    return (1.0 - float(goal_reached))*dist2


###############################################################################
# 5) The Key: MPPI "rollout" kernel WITH region-of-attraction penalty
###############################################################################
@cuda.jit
def rollout_kernel(
    x0_d,            # (12,) initial state
    xgoal_d,         # (12,) target state
    dt,              # float
    mass,            # float
    dist_weight,     # float
    roa_radius_sq,   # float -> region-of-attraction squared
    goal_tolerance_sq, # float
    noise_samples_d, # (num_rollouts, num_steps, 6)
    u_cur_d,         # (num_steps, 6)
    costs_d          # (num_rollouts,)
):
    """
    One block per rollout. We'll do a single-thread block for clarity.
    """
    bid = cuda.blockIdx.x
    # We'll do everything in that block
    # local copy of x state
    x_curr = cuda.local.array(12, numba.float32)
    for i in range(12):
        x_curr[i] = x0_d[i]

    n_steps = u_cur_d.shape[0]
    costs_d[bid] = 0.0
    goal_reached = False

    for t in range(n_steps):
        # add noise
        Fx = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
        Fy = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
        Fz = u_cur_d[t, 2] + noise_samples_d[bid, t, 2]
        Mx = u_cur_d[t, 3] + noise_samples_d[bid, t, 3]
        My = u_cur_d[t, 4] + noise_samples_d[bid, t, 4]
        Mz = u_cur_d[t, 5] + noise_samples_d[bid, t, 5]

        # forward simulate
        ctrl = (Fx, Fy, Fz, Mx, My, Mz)
        hex_dynamics_inplace(x_curr, ctrl, dt, mass)

        # dist to final
        dx = xgoal_d[0] - x_curr[0]
        dy = xgoal_d[1] - x_curr[1]
        dz = xgoal_d[2] - x_curr[2]
        d2 = dx*dx + dy*dy + dz*dz
        # stage cost
        costs_d[bid] += stage_cost(d2, dist_weight)

        # region-of-attraction check
        if d2 > roa_radius_sq:
            # big penalty if we leave stable region
            costs_d[bid] += 1e6
            break

        if d2 < goal_tolerance_sq:
            goal_reached = True
            break

    # add terminal cost
    costs_d[bid] += term_cost(d2, goal_reached)


###############################################################################
# 6) MPPI Weight-Update Kernel
###############################################################################
@cuda.jit
def update_useq_kernel(
    costs_d,                 # shape (num_rollouts,)
    noise_samples_d,         # shape (num_rollouts, num_steps, 6)
    weights_d,               # shape (num_rollouts,)
    u_cur_d,                 # shape (num_steps,6)
    lambda_weight, 
    num_rollouts
):
    """
    Single-block approach for simplicity. We'll do a basic cost-min reduction, 
    then exponent weights, then atomic adds to update u_cur_d.

    Each thread handles a slice of rollouts => partial reduce => atomic ops.
    """
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    # 1) find minimal cost for normalization
    min_cost = 1e30
    for i in range(tid, num_rollouts, block_size):
        c = costs_d[i]
        if c < min_cost:
            min_cost = c
    # parallel reduce min_cost
    sm = cuda.shared.array(1, numba.float32)
    sm[0] = min_cost
    cuda.syncthreads()

    # reduce across threads
    if tid == 0:
        # just do a naive loop to combine
        for t2 in range(1, block_size):
            # pretend the other threads stored their local min in sm[t2], 
            # or do an atomic approach. We'll keep it simple for demonstration 
            pass
    cuda.syncthreads()
    # let's assume min_cost is in sm[0]

    # 2) compute weights
    for i in range(tid, num_rollouts, block_size):
        w = math.exp(-1.0 / lambda_weight * (costs_d[i] - sm[0]))
        weights_d[i] = w
    cuda.syncthreads()

    # 3) sum of weights for normalization
    local_sum = 0.0
    for i in range(tid, num_rollouts, block_size):
        local_sum += weights_d[i]
    # reduce sum
    # store local sum in shared memory
    tmp_sum = cuda.shared.array(1, numba.float32)
    tmp_sum[0] = 0.0
    cuda.syncthreads()

    # atomic add
    cuda.atomic.add(tmp_sum, 0, local_sum)
    cuda.syncthreads()

    # normalize
    total_weight = tmp_sum[0]
    for i in range(tid, num_rollouts, block_size):
        weights_d[i] /= total_weight
    cuda.syncthreads()

    # 4) update control
    # first zero out u_cur_d (we do it once)
    if tid==0:
        for ts in range(u_cur_d.shape[0]):
            for c_ in range(u_cur_d.shape[1]):
                u_cur_d[ts,c_] = 0.0
    cuda.syncthreads()

    # do weighted sum
    for i in range(tid, num_rollouts, block_size):
        w = weights_d[i]
        for ts in range(u_cur_d.shape[0]):
            # the noise contributed is noise_samples_d[i, ts, c_]
            # we do atomic add
            cuda.atomic.add(u_cur_d, (ts, 0), w* noise_samples_d[i, ts, 0])
            cuda.atomic.add(u_cur_d, (ts, 1), w* noise_samples_d[i, ts, 1])
            cuda.atomic.add(u_cur_d, (ts, 2), w* noise_samples_d[i, ts, 2])
            cuda.atomic.add(u_cur_d, (ts, 3), w* noise_samples_d[i, ts, 3])
            cuda.atomic.add(u_cur_d, (ts, 4), w* noise_samples_d[i, ts, 4])
            cuda.atomic.add(u_cur_d, (ts, 5), w* noise_samples_d[i, ts, 5])


###############################################################################
# 7) MPPI_Numba Class with Region-of-Attraction Enforcement
###############################################################################
class MPPI_Numba:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_steps = cfg.num_steps
        self.num_control_rollouts = cfg.num_control_rollouts
        self.num_controls = cfg.num_controls
        self.num_states = cfg.num_states
        self.dt = cfg.dt
        self.seed = cfg.seed

        # region-of-attraction radius for short-horizon MPC
        self.roa_radius = 2.0       # can be tuned
        self.goal_tolerance = 0.05  # can be tuned

        # device arrays
        self.noise_samples_d = None
        self.costs_d = None
        self.weights_d = None
        self.u_cur_d = None
        self.rng_states_d = None

        # store initial guess for controls
        mass = 7.0
        g = 9.81
        self.u_seq0 = np.zeros((self.num_steps,self.num_controls),dtype=np.float32)
        self.u_seq0[:,2] = mass*g  # hover

        # We'll do params like x0, xgoal, dist_weight, etc. in a dict
        self.params = {}
        self.initialized = False
        self._init_device()

    def _init_device(self):
        self.noise_samples_d = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
        self.costs_d = cuda.device_array((self.num_control_rollouts,), dtype=np.float32)
        self.weights_d = cuda.device_array((self.num_control_rollouts,), dtype=np.float32)
        self.u_cur_d = cuda.to_device(self.u_seq0)
        self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts*self.num_steps, seed=self.seed)
        self.initialized = True

    def set_params(self, param_dict):
        """
        param_dict might contain:
         - 'x0': (12,) array
         - 'xgoal': (12,) array
         - 'dist_weight': float
         - 'lambda_weight': float
         - 'num_opt': int
         - 'mass': float
        """
        self.params = copy.deepcopy(param_dict)

    def solve(self):
        if not self.initialized:
            print("Device not ready.")
            return self.u_seq0
        return self._solve_main()

    def _solve_main(self):
        # read from self.params
        x0   = self.params.get('x0', np.zeros(12, dtype=np.float32))
        xgoal= self.params.get('xgoal', np.zeros(12, dtype=np.float32))
        dt_  = self.params.get('dt', self.dt)
        mass = self.params.get('mass', 7.0)
        dist_weight = self.params.get('dist_weight', 2000.0)
        lam_w       = self.params.get('lambda_weight', 10.0)
        num_opt     = self.params.get('num_opt', 5)

        # device copies
        x0_d        = cuda.to_device(x0.astype(np.float32))
        xgoal_d     = cuda.to_device(xgoal.astype(np.float32))
        dt_d        = np.float32(dt_)
        mass_d      = np.float32(mass)
        dist_weight_d = np.float32(dist_weight)
        roa_sq      = np.float32(self.roa_radius*self.roa_radius)
        goal_tol_sq = np.float32(self.goal_tolerance*self.goal_tolerance)

        # main optimization loop
        block_rollouts = (self.num_control_rollouts,1)
        grid_update = (1,1)
        threads_update = (128,1)   # arbitrary

        for _ in range(num_opt):
            # 1) sample noise
            self._sample_noise()
            # 2) rollout
            rollout_kernel[block_rollouts, 1](
                x0_d,
                xgoal_d,
                dt_d,
                mass_d,
                dist_weight_d,
                roa_sq,
                goal_tol_sq,
                self.noise_samples_d,
                self.u_cur_d,
                self.costs_d
            )
            # 3) update
            update_useq_kernel[grid_update, threads_update](
                self.costs_d,
                self.noise_samples_d,
                self.weights_d,
                self.u_cur_d,
                lam_w,
                self.num_control_rollouts
            )

        # return final
        return self.u_cur_d.copy_to_host()

    def _sample_noise(self):
        """
        Just do a normal sampling for each step, each control dimension.
        """
        block = (self.num_control_rollouts, 1)
        thread = (self.num_steps, 1)
        _sample_noise_kernel[block, thread](
            self.rng_states_d,
            self.noise_samples_d
        )

    def shift_and_update(self, new_x0, new_u, num_shifts=1):
        """
        SHIFT the solution by 'num_shifts' steps, then update self.params['x0']
        """
        self.params['x0'] = new_x0.copy()
        # shift array:
        if num_shifts< self.num_steps:
            shifted = new_u.copy()
            shifted[:-num_shifts,:] = shifted[num_shifts:,:]
            # optional zero tail
            shifted[-num_shifts:,:] = 0.0
            self.u_cur_d = cuda.to_device(shifted.astype(np.float32))


@cuda.jit
def _sample_noise_kernel(rng_states, noise_samples_d):
    """
    Each block = 1 rollout. Each thread = 1 step. 
    We'll do basic normal(0,0.5) for demonstration.
    """
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    # we have 6 controls
    # we just do standard normal(0, 1) scaled by 0.5 for demonstration
    scale = 0.5
    if tid < noise_samples_d.shape[1]:
        for c_ in range(noise_samples_d.shape[2]):
            val = xoroshiro128p_normal_float32(rng_states, bid*noise_samples_d.shape[1] + tid)
            noise_samples_d[bid, tid, c_] = scale* val


###############################################################################
# 8) MAIN DEMO
###############################################################################
def main():
    # 1) Build config
    cfg = Config(
        T=2.0,
        dt=0.3,
        num_control_rollouts=512,
        num_controls=6,
        num_states=12,
        num_vis_state_rollouts=1,
        seed=123
    )

    # 2) Build short-horizon MPC (placeholder)
    mpc_params = {
        "mass": 7.0,
        "gravity": 9.81,
    }
    my_mpc = MPC(mpc_params)

    # 3) Build MPPI
    mppi = MPPI_Numba(cfg)
    # initial state
    x0 = np.zeros(12, dtype=np.float32)
    # desired final
    xgoal = np.array([1.0, -1.0, 2.0, 0,0,0, 0,0,0, 0,0,0], dtype=np.float32)

    # MPPI parameters
    mppi_params = {
        "x0": x0,
        "xgoal": xgoal,
        "dt": 0.3,
        "mass": 7.0,
        "dist_weight": 2000.0,
        "lambda_weight": 10.0,
        "num_opt": 8
    }
    mppi.set_params(mppi_params)

    # Simulation
    max_steps = 30
    xhist = []
    xhist.append(x0.copy())
    uhist = []

    # we'll do a naive loop
    for t in range(max_steps):
        # 1) MPPI solve => get best controls
        useq = mppi.solve()  # shape (num_steps,6)
        u_first = useq[0,:].copy()

        # 2) Use the low-level MPC on the "sub-target"
        #    For demonstration, let's just do "a short forward-sim" 
        sub_target = forward_sim_mppi_target(useq, xhist[-1], dt=0.3)
        # then call the short-horizon MPC
        u_mpc = my_mpc.compute_control(xhist[-1], sub_target, np.zeros(6), 0.01)

        # 3) Decide which control we actually apply
        #    e.g. if we prefer the MPPI's direct approach => use u_first
        #    or if we prefer the short-horizon => use u_mpc
        # here we pick MPPI approach
        u_apply = u_first

        # 4) Simulate the real state forward
        x_next = simple_rk4(xhist[-1], u_apply, dt=0.3)
        xhist.append(x_next.copy())
        uhist.append(u_apply.copy())

        # 5) shift the MPPI solution
        mppi.shift_and_update(x_next, useq, num_shifts=1)

        print(f"Step {t}, x=({x_next[0]:.2f}, {x_next[1]:.2f}, {x_next[2]:.2f})")

    # Plot
    xhist_arr = np.array(xhist)
    time_arr = np.arange(len(xhist_arr))*0.3
    plt.figure()
    plt.plot(time_arr, xhist_arr[:,0], label="x")
    plt.plot(time_arr, xhist_arr[:,1], label="y")
    plt.plot(time_arr, xhist_arr[:,2], label="z")
    plt.legend()
    plt.title("Position")
    plt.show()


def forward_sim_mppi_target(useq, state, dt=0.3):
    """
    Optionally forward-simulate the first few steps of MPPI control to get 
    a sub-target for the low-level MPC. 
    Here we just do one step for demonstration.
    """
    # do a single step of size dt
    ctrl = useq[0,:]
    next_st = simple_rk4(state, ctrl, dt)
    return next_st


def simple_rk4(state, ctrl, dt):
    """
    Minimal RK4 of 12D state with a 6D control. 
    Using hex_dynamics_inplace device code, but we do it in python for brevity.
    """
    mass = 7.0
    # We'll define a quick python version for demonstration
    def f(x, u):
        # copy x
        xloc = x.copy()
        hex_dynamics_inplace_python(xloc, u, mass)
        return xloc - x  # difference

    k1 = f(state, ctrl)*dt
    k2 = f(state + 0.5*k1, ctrl)*dt
    k3 = f(state + 0.5*k2, ctrl)*dt
    k4 = f(state + k3, ctrl)*dt
    return state + (k1 + 2*k2 + 2*k3 + k4)/6.0


def hex_dynamics_inplace_python(x, u, mass):
    """
    A python version of the same logic as hex_dynamics_inplace, for CPU testing
    """
    g=9.81
    Ixx,Iyy,Izz=0.115125971,0.116524229,0.230387752
    # pos
    x[0]+= x[3]
    x[1]+= x[4]
    x[2]+= x[5]

    # vel
    Fx, Fy, Fz = u[0],u[1],u[2]
    # ignoring orientation for gravity => simplify
    x[3]+= (Fx/mass - g*(0.0))
    x[4]+= (Fy/mass - g*(0.0))
    x[5]+= (Fz/mass - g*(1.0))

    # orientation
    x[6]+= x[9]
    x[7]+= x[10]
    x[8]+= x[11]
    # angular
    Mx, My, Mz = u[3],u[4],u[5]
    p_,q_,r_= x[9],x[10],x[11]
    x[9]+= (1./Ixx)*(Mx + (Iyy - Izz)*q_*r_)
    x[10]+=(1./Iyy)*(My + (Izz - Ixx)*p_*r_)
    x[11]+=(1./Izz)*(Mz + (Ixx - Iyy)*p_*q_)


if __name__=="__main__":
    main()
