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


# Information about your GPU
gpu = cuda.get_current_device()
max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
max_blocks = gpu.MAX_GRID_DIM_X
max_rec_blocks = rec_max_control_rollouts = int(1e6) # Though theoretically limited by max_blocks on GPU
rec_min_control_rollouts = 100

CONTACT_NORMAL = np.array([-1, 0, 0], dtype=np.float32)

class Config:
  
  """ Configurations that are typically fixed throughout execution. """
  
  def __init__(self, 
               T=0.5, # Horizon (s)
               dt=0.02, # Length of each step (s)
               num_control_rollouts=1024, # Number of control sequences
               num_controls = 6,
               num_states = 12,
               num_vis_state_rollouts=20, # Number of visualization rollouts
               seed=1):
    
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    self.max_threads_per_block = max_threads_per_block # save just in case
    self.num_controls = num_controls
    self.num_states = num_states

    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0

    
    # Number of control rollouts are currently limited by the number of blocks
    self.num_control_rollouts = num_control_rollouts
    if self.num_control_rollouts > rec_max_control_rollouts:
      self.num_control_rollouts = rec_max_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended max number of {}. (Max={})".format(
        rec_max_control_rollouts, max_blocks))
    elif self.num_control_rollouts < rec_min_control_rollouts:
      self.num_control_rollouts = rec_min_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended min number of {}. (Recommended max={})".format(
        rec_min_control_rollouts, rec_max_control_rollouts))
    
    # For visualizing state rollouts
    self.num_vis_state_rollouts = num_vis_state_rollouts
    self.num_vis_state_rollouts = min([self.num_vis_state_rollouts, self.num_control_rollouts])
    self.num_vis_state_rollouts = max([1, self.num_vis_state_rollouts])

    print("num_steps: ", self.num_steps)

DEFAULT_OBS_COST = 1e3
DEFAULT_DIST_WEIGHT = 10

# Stage costs (device function)
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def stage_cost(dist2, dist_weight):
  return dist_weight*dist2 # squared term makes the robot move faster

# Terminal costs (device function)
@cuda.jit('float32(float32, boolean)', device=True, inline=True)
def term_cost(dist2, goal_reached):
  return (1-np.float32(goal_reached))*dist2

@cuda.jit(device=True, fastmath=True)
def dynamics_update_lcp_contact_force(x, u, dt, contact_normal, inertia_mass, plane, f_contact_out=None):
    """
    Single-plane implicit-compliant contact model (backward-Euler spring–damper + Coulomb friction).

    Stable for large dt and avoids LCP jitter.

    x: state [px,py,pz, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]
    u: control [Fx_b, Fy_b, Fz_b, Mx, My, Mz]
    plane: [A,B,C,D]
    inertia_mass: [Ixx,Iyy,Izz,m]
    f_contact_out: optional (3,) world-frame contact force
    """
    # Parameters
    arm_length = 1.2
    mu = 0.01
    k_spring = 200.0   # N/m
    c_damp   = 20.0     # N·s/m

    # Unpack inertia
    Ixx, Iyy, Izz, m = inertia_mass[0], inertia_mass[1], inertia_mass[2], inertia_mass[3]
    g = 9.81

    # Unpack state
    px, py, pz = x[0], x[1], x[2]
    vx, vy, vz = x[3], x[4], x[5]
    phi, th, psi = x[6], x[7], x[8]
    wx_b, wy_b, wz_b = x[9], x[10], x[11]

    # Rotation Z-Y-X
    sphi, cphi = math.sin(phi), math.cos(phi)
    sth, cth   = math.sin(th),   math.cos(th)
    spsi, cpsi = math.sin(psi),  math.cos(psi)
    R00 = cth*cpsi; R01 = cth*spsi; R02 = -sth
    R10 = sphi*sth*cpsi - cphi*spsi
    R11 = sphi*sth*spsi + cphi*cpsi
    R12 = sphi*cth
    R20 = cphi*sth*cpsi + sphi*spsi
    R21 = cphi*sth*spsi - sphi*cpsi
    R22 = cphi*cth

    # Body->world force
    Fx_w = R00*u[0] + R01*u[1] + R02*u[2]
    Fy_w = R10*u[0] + R11*u[1] + R12*u[2]
    Fz_w = R20*u[0] + R21*u[1] + R22*u[2] - m*g

    # Integrate flight
    px += dt*vx; py += dt*vy; pz += dt*vz
    vx += dt*Fx_w/m; vy += dt*Fy_w/m; vz += dt*Fz_w/m

    # Integrate orientation
    phi_dot = wx_b + sphi*math.tan(th)*wy_b + cphi*math.tan(th)*wz_b
    th_dot  = cphi*wy_b - sphi*wz_b
    psi_dot = (sphi*wy_b + cphi*wz_b)/cth
    phi += dt*phi_dot; th += dt*th_dot; psi += dt*psi_dot

    # Integrate angular velocity
    wx_dot = (u[3] + (Iyy-Izz)*wy_b*wz_b)/Ixx
    wy_dot = (u[4] + (Izz-Ixx)*wx_b*wz_b)/Iyy
    wz_dot = (u[5] + (Ixx-Iyy)*wx_b*wy_b)/Izz
    wx_b += dt*wx_dot; wy_b += dt*wy_dot; wz_b += dt*wz_dot

    # End-effector pos & vel
    ee_x = px + R00*arm_length; ee_y = py + R10*arm_length; ee_z = pz + R20*arm_length
    w_wx = R00*wx_b + R01*wy_b + R02*wz_b
    w_wy = R10*wx_b + R11*wy_b + R12*wz_b
    w_wz = R20*wx_b + R21*wy_b + R22*wz_b
    r_wx = R00*arm_length; r_wy = R10*arm_length; r_rz = R20*arm_length
    v_ex = vx + (w_wy*r_rz - w_wz*r_wy)
    v_ey = vy + (w_wz*r_wx - w_wx*r_rz)
    v_ez = vz + (w_wx*r_wy - w_wy*r_wx)

    # Compute penetration
    A, B, C = plane[0], plane[1], plane[2]
    D = plane[3]
    norm_n = math.sqrt(A*A + B*B + C*C) + 1e-9
    nx, ny, nz = A/norm_n, B/norm_n, C/norm_n
    pen = (A*ee_x + B*ee_y + C*ee_z + D) / norm_n

    # Contact force
    Fx_c = 0.0; Fy_c = 0.0; Fz_c = 0.0
    if pen > 0.0001:
        # normal velocity
        v_n = v_ex*nx + v_ey*ny + v_ez*nz
        # solve implicit compliance magnitude
        Fn_mag = (k_spring*pen + c_damp*v_n) / (1.0 + (c_damp*dt)/m)
        # ensure non-negative magnitude
        if Fn_mag < 0.0:
            Fn_mag = 0.0
        # normal force vector (opposite penetration normal)
        Fx_n = -Fn_mag * nx
        Fy_n = -Fn_mag * ny
        Fz_n = -Fn_mag * nz
        # Coulomb friction
        vt_x = v_ex - v_n*nx; vt_y = v_ey - v_n*ny; vt_z = v_ez - v_n*nz
        vt_mag = math.sqrt(vt_x*vt_x + vt_y*vt_y + vt_z*vt_z) + 1e-9
        Ft_mag = mu * Fn_mag
        dx, dy, dz = -vt_x/vt_mag, -vt_y/vt_mag, -vt_z/vt_mag
        Fx_c = Fx_n + Ft_mag*dx
        Fy_c = Fy_n + Ft_mag*dy
        Fz_c = Fz_n + Ft_mag*dz
        # apply contact forces
        vx += (Fx_c * dt) / m
        vy += (Fy_c * dt) / m
        vz += (Fz_c * dt) / m
        # position correction
        px -= pen * nx; py -= pen * ny; pz -= pen * nz

    # write-back state
    x[0], x[1], x[2] = px, py, pz
    x[3], x[4], x[5] = vx, vy, vz
    x[6], x[7], x[8] = phi, th, psi
    x[9], x[10], x[11] = wx_b, wy_b, wz_b
    if f_contact_out is not None:
        f_contact_out[0], f_contact_out[1], f_contact_out[2] = Fx_c, Fy_c, Fz_c




class MPPI_Numba(object):
  
  """ 
  Implementation of Information theoretic MPPI by Williams et. al. 
  Alg 2. in https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf


  Controller object that initializes GPU memory and runs MPPI on GPU via numba. 
  
  Typical workflow: 
    1. Initialize object with config that allows pre-initialization of GPU memory
    2. reset()
    3. set_params(mppi_params) based on problem instance
    4. solve(), which returns optimized control sequence
    5. get_state_rollout() for visualization
    6. shift_and_update(next_state, optimal_u_sequence, num_shifts=1)
    7. Repeat from 2 if params have changed
  """

  def __init__(self, cfg):

    # Fixed configs
    self.cfg = cfg
    self.T = cfg.T
    self.dt = cfg.dt
    self.num_steps = cfg.num_steps
    self.num_control_rollouts = cfg.num_control_rollouts
    self.num_controls = cfg.num_controls
    self.num_states = cfg.num_states

    self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
    self.seed = cfg.seed

    # Basic info 
    self.max_threads_per_block = cfg.max_threads_per_block

    # Initialize reuseable device variables
    self.noise_samples_d = None
    self.u_cur_d = None
    self.u_prev_d = None
    self.costs_d = None
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient

    # Other task specific params
    self.last_noise_d = None # keep last noise for ou process noise samping
    # OU params
    self.use_ou = False #
    self.theta = 2  # OU process theta
    self.mu = 0.0  # OU process mean
    self.sigma = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.03])*0.2
    self.delta_t = self.cfg.dt  # Time step, already defined in Config
    self.ou_alpha = 0.7
    self.ou_scale = 1
    self.d_ou_scale = 0.5
    self.sys_noise = np.array([0.1, 0.1, 0.1, 0.001, 0.001, 0.001])
    self.umin = np.array([-20, -20, -40, -0.1, -0.1, -0.1])  # Example minimum control values
    self.umax = np.array([20, 20, 40, 0.1, 0.1, 0.1])  # Example maximum control values
    self.last_controls = np.zeros((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
    self.last_controls_d = cuda.to_device(self.last_controls.astype(np.float32))
    # other params , A, B, C, D, ABC_sq, contact_normal_sq, contact_normal
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
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, self.num_controls), dtype=np.float32)
    mass = 7.00
    g = 9.81
    self.u_seq0[:, 2] = mass * g  # Set hover thrust in the z-direction
    self.params = None
    self.params_set = False

    self.u_prev_d = None

    self.last_noise_d = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
    
    # Initialize all fixed-size device variables ahead of time. (Do not change in the lifetime of MPPI object)
    self.init_device_vars_before_solving()


  def init_device_vars_before_solving(self):

    if not self.device_var_initialized:
      t0 = time.time()
      
      self.noise_samples_d = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32) # to be sampled collaboratively via GPU
      self.u_cur_d = cuda.to_device(self.u_seq0) 
      self.u_prev_d = cuda.to_device(self.u_seq0) 
      self.costs_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.weights_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts*self.num_steps, seed=self.seed)
      
      self.state_rollout_batch_d = cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, self.num_states), dtype=np.float32)
      
      self.device_var_initialized = True
      print("MPPI planner has initialized GPU memory after {} s".format(time.time()-t0))

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
    """Entry point for different algoritims"""
    
    if not self.check_solve_conditions():
      print("MPPI solve condition not met. Cannot solve. Return")
      return
    
    return self.solve_with_nominal_dynamics()

  def change_goal(self, goal):
    self.params['xgoal'] = goal

  def move_mppi_task_vars_to_device(self):
    vrange_d = cuda.to_device(self.params['vrange'].astype(np.float32))
    wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
    xgoal_d = cuda.to_device(self.params['xgoal'].astype(np.float32))
    fgoal_d = cuda.to_device(self.params['fgoal'].astype(np.float32))
    plane_d = cuda.to_device(self.params['plane'].astype(np.float32))
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])
    lambda_weight_d = np.float32(self.params['lambda_weight'])
    u_std_d = cuda.to_device(self.params['u_std'].astype(np.float32))
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])
    cost_weights_d = cuda.to_device(self.params['weights'].astype(np.float32))
    inertia_mass_d = cuda.to_device(self.params['inertia_mass'].astype(np.float32))

    if "obstacle_positions" in self.params:
      obs_pos_d = cuda.to_device(self.params['obstacle_positions'].astype(np.float32))
    else:
      obs_pos_d = np.array([[1e5,1e5]], dtype=np.float32) # dummy value, else numba panics : (
    if "obstacle_radius" in self.params:
      obs_r_d = cuda.to_device(self.params['obstacle_radius'].astype(np.float32))
    else:
      obs_r_d = np.array([0], dtype=np.float32) # dummy value, else numba panics : (

    obs_cost_d = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params 
                                     else self.params['obs_penalty'])
    return vrange_d, wrange_d, xgoal_d, fgoal_d, plane_d,\
           goal_tolerance_d, lambda_weight_d, \
           u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d, \
           cost_weights_d, inertia_mass_d


  def solve_with_nominal_dynamics(self):
    """
    Launch GPU kernels that use nominal dynamics but adjsuts cost function based on worst-case linear speed.
    """
    
    vrange_d, wrange_d, xgoal_d, fgoal_d, plane_d, goal_tolerance_d, lambda_weight_d, \
           u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d, cost_weights_d, inertia_mass_d = self.move_mppi_task_vars_to_device()
   
    dist_to_goal_d = cuda.device_array(6, dtype=np.float32)  # Add distance to goal for each control
    coef_dist_to_goal = np.array([1, 1, 5, 0.03, 0.03, 0.03], dtype=np.float32)*0.1  # Coefficients for distance scaling

    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    

    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      if self.use_ou:
        # Scale the std by distance
        dist_to_goal = (self.params['xgoal'][:6] - self.params['x0'][:6])**2
        # Call OU noise sampling kernel
        self.sample_noise_ou_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d,
            self.theta,
            self.mu,
            self.sigma,
            self.dt,
            self.noise_samples_d)

      else:
        dist_to_goal = np.abs(self.params['xgoal'][:6] - self.params['x0'][:6])
        u_std_scaled = np.minimum(u_std_d, coef_dist_to_goal * dist_to_goal)  # Scale noise std by distance (TODO: use this indstead of u_std_d and tune)
        self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
        
      # print(f'u_curr_d: [{self.u_cur_d[0,0]}, {self.u_cur_d[0,1]}, {self.u_cur_d[0,2]}, {self.u_cur_d[0,3]}, {self.u_cur_d[0,4]}, {self.u_cur_d[0,5]}]')
      # Rollout and compute mean or cvar
      self.rollout_numba[self.num_control_rollouts, 1](
        inertia_mass_d,
        vrange_d,
        wrange_d,
        xgoal_d,
        fgoal_d,
        plane_d,
        obs_cost_d, 
        obs_pos_d, 
        obs_r_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        x0_d,
        dt_d,
        dist_weight,
        cost_weights_d,
        self.noise_samples_d,
        self.u_cur_d,
        # results
        self.costs_d
      )
      self.u_prev_d = self.u_cur_d

      # Compute cost and update the optimal control on device
      self.update_useq_numba[1, 32](
        lambda_weight_d, 
        self.costs_d, 
        self.noise_samples_d, 
        self.weights_d, 
        vrange_d,
        wrange_d,
        self.u_cur_d
      )

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
    """
    Generate state sequences based on the current optimal control sequence.
    """

    assert self.params_set, "MPPI parameters are not set"

    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot run mppi.")
      return
    
    # Move things to GPU
    vrange_d = cuda.to_device(self.params['vrange'].astype(np.float32))
    wrange_d = cuda.to_device(self.params['wrange'].astype(np.float32))
    x0_d = cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])

    self.get_state_rollout_across_control_noise[self.num_vis_state_rollouts, 1](
        self.state_rollout_batch_d, # where to store results
        x0_d, 
        dt_d,
        self.noise_samples_d,
        vrange_d,
        wrange_d,
        self.u_prev_d,
        self.u_cur_d,
        )
    
    return self.state_rollout_batch_d.copy_to_host()


  """GPU kernels from here on"""
  @staticmethod
  @cuda.jit(fastmath=True)
  def rollout_numba(
          inertia_mass_d,
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          fgoal_d,
          plane_d,
          obs_cost_d, 
          obs_pos_d, 
          obs_r_d,
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          x0_d, 
          dt_d,
          dist_weight_d,
          cost_weights_d,
          noise_samples_d,
          u_cur_d,
          costs_d):
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """

    # Get block id and thread id
    bid = cuda.blockIdx.x   # index of block
    tid = cuda.threadIdx.x  # index of thread within a block
    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = cuda.local.array(12, numba.float32)
    for i in range(12): 
      x_curr[i] = x0_d[i]
    timesteps = len(u_cur_d)
    goal_reached = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9
    u_nom =  cuda.local.array(6, numba.float32)

    # Initialize previous control input
    u_prev = cuda.local.array(6, numba.float32)
    for i in range(6):
      u_prev[i] = u_cur_d[0, i]

    # printed=False
    for t in range(timesteps):
     
      # Nominal noisy control
      u_nom[0] = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
      u_nom[1] = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      u_nom[2] = u_cur_d[t, 2] + noise_samples_d[bid, t, 2]
      u_nom[3] = u_cur_d[t, 3] + noise_samples_d[bid, t, 3]
      u_nom[4] = u_cur_d[t, 4] + noise_samples_d[bid, t, 4]
      u_nom[5] = u_cur_d[t, 5] + noise_samples_d[bid, t, 5]

      # TODO: implement control limits  
      u_noisy = u_nom
      # u_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      
      cf = cuda.local.array(3, float32)  # local array to hold contact force
      # Forward simulate
      dynamics_update_lcp_contact_force(x_curr, u_noisy, dt_d, CONTACT_NORMAL, inertia_mass_d, plane_d, cf)
      # If else statements will be expensive
      dist_to_goal2 = cost_weights_d[0]*((xgoal_d[0]-x_curr[0])**2) + cost_weights_d[1]*((xgoal_d[1]-x_curr[1])**2) + cost_weights_d[2]*((xgoal_d[2]-x_curr[2])**2) \
                    + cost_weights_d[3]*((xgoal_d[3]-x_curr[3])**2) + cost_weights_d[4]*((xgoal_d[4]-x_curr[4])**2) + cost_weights_d[5]*((xgoal_d[5]-x_curr[5])**2)\
                    + cost_weights_d[6]*((xgoal_d[6]-x_curr[6])**2) + cost_weights_d[7]*((xgoal_d[7]-x_curr[7])**2) + cost_weights_d[8]*((xgoal_d[8]-x_curr[8])**2)\
                    + cost_weights_d[9]*((xgoal_d[9]-x_curr[9])**2) + cost_weights_d[10]*((xgoal_d[10]-x_curr[10])**2) + cost_weights_d[11]*(xgoal_d[11]-x_curr[11])**2\
                    + cost_weights_d[12]*(50*(u_nom[0]**2) + (u_nom[1]**2) + ((u_nom[2] - inertia_mass_d[3]*9.81)**2))\
                    + cost_weights_d[13]*((u_nom[3]**2) + (u_nom[4]**2) + (u_nom[5]**2))\
                    + cost_weights_d[17] * ((cf[0] + fgoal_d[0]) ** 2) + cost_weights_d[18] * ((cf[2] + fgoal_d[2]) ** 2) + cost_weights_d[19] * ((cf[2] + fgoal_d[2]) ** 2) 
                    
      costs_d[bid]+= stage_cost(dist_to_goal2, dist_weight_d)

      if dist_to_goal2<= goal_tolerance_d2:
        goal_reached = True
        break
    # Accumulate terminal cost 
    costs_d[bid] += cost_weights_d[16] * term_cost(dist_to_goal2, goal_reached)
    # Add Control cost 
    for t in range(timesteps):
      costs_d[bid] += cost_weights_d[14]*lambda_weight_d*(
              (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1] + (u_cur_d[t,2]/(u_std_d[2]**2))*noise_samples_d[bid, t, 2]\
                 + cost_weights_d[15]*((u_cur_d[t,3]/(u_std_d[3]**2))*noise_samples_d[bid, t, 3] + (u_cur_d[t,4]/(u_std_d[4]**2))*noise_samples_d[bid, t, 4] + (u_cur_d[t,5]/(u_std_d[5]**2))*noise_samples_d[bid, t, 5]))
  @staticmethod
  @cuda.jit(fastmath=True)
  def update_useq_numba(
        lambda_weight_d,
        costs_d,
        noise_samples_d,
        weights_d,
        vrange_d,
        wrange_d,
        u_cur_d):
    """
    GPU kernel that updates the optimal control sequence based on previously evaluated cost values.
    Assume that the function is invoked as update_useq_numba[1, NUM_THREADS], with one block and multiple threads.
    """

    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    numel = len(noise_samples_d)
    gap = int(math.ceil(numel / num_threads))

    # Find the minimum value via reduction
    starti = min(tid*gap, numel)
    endi = min(starti+gap, numel)
    if starti<numel:
      weights_d[starti] = costs_d[starti]
    for i in range(starti, endi):
      weights_d[starti] = min(weights_d[starti], costs_d[i])
    cuda.syncthreads()

    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        weights_d[starti] = min(weights_d[starti], weights_d[starti + s])
      s *= 2
      cuda.syncthreads()

    beta = weights_d[0]
    
    # Compute weight
    for i in range(starti, endi):
      weights_d[i] = math.exp(-1./lambda_weight_d*(costs_d[i]-beta))
    cuda.syncthreads()

    # Normalize
    # Reuse costs_d array
    for i in range(starti, endi):
      costs_d[i] = weights_d[i]
    cuda.syncthreads()
    for i in range(starti+1, endi):
      costs_d[starti] += costs_d[i]
    cuda.syncthreads()
    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        costs_d[starti] += costs_d[starti + s]
      s *= 2
      cuda.syncthreads()

    for i in range(starti, endi):
      weights_d[i] /= costs_d[0]
    cuda.syncthreads()
    
    # update the u_cur_d
    timesteps = len(u_cur_d)
    for t in range(timesteps):
      for i in range(starti, endi):
        cuda.atomic.add(u_cur_d, (t, 0), weights_d[i]*noise_samples_d[i, t, 0])
        cuda.atomic.add(u_cur_d, (t, 1), weights_d[i]*noise_samples_d[i, t, 1])
        cuda.atomic.add(u_cur_d, (t, 2), weights_d[i]*noise_samples_d[i, t, 2])
        cuda.atomic.add(u_cur_d, (t, 3), weights_d[i]*noise_samples_d[i, t, 3])
        cuda.atomic.add(u_cur_d, (t, 4), weights_d[i]*noise_samples_d[i, t, 4])
        cuda.atomic.add(u_cur_d, (t, 5), weights_d[i]*noise_samples_d[i, t, 5])
    cuda.syncthreads()

    # Blocks crop the control together
    tgap = int(math.ceil(timesteps / num_threads))
    starti = min(tid*tgap, timesteps)
    endi = min(starti+tgap, timesteps)
    # for ti in range(starti, endi):
    #   # u_cur_d[ti, 0] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 0]))
    #   # u_cur_d[ti, 1] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 1]))
    #   # u_cur_d[ti, 2] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 2]))
    #   # u_cur_d[ti, 3] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 3]))
    #   # u_cur_d[ti, 4] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 4]))
    #   # u_cur_d[ti, 5] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 5]))
    #   # u_cur_d[ti, 0] = max(-10, min(10, u_cur_d[ti, 0]))
    #   # u_cur_d[ti, 1] = max(-10, min(10, u_cur_d[ti, 1]))
    #   # u_cur_d[ti, 2] = max(0, min(60, u_cur_d[ti, 2]))
    #   u_cur_d[ti, 3] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 3]))
    #   u_cur_d[ti, 4] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 4]))
    #   u_cur_d[ti, 5] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 5]))



  @staticmethod
  @cuda.jit(fastmath=True)
  def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
      """
      Generate noise samples with linearly interpolated variance for the first half
      of the horizon and constant variance for the second half.
      """
      block_id = cuda.blockIdx.x
      thread_id = cuda.threadIdx.x
      abs_thread_id = cuda.grid(1)
      num_timesteps = noise_samples_d.shape[1]
      num_controls = noise_samples_d.shape[2]
  
      # denom = 1
      # for i in range(num_controls):
      #     for t in range(num_timesteps):
      #         # Determine scaling for variance
      #         if t < num_timesteps // 2:  # First half
      #             scale = (t / (num_timesteps // 2)) * (denom - 1) / denom + 1 / denom  # Interpolation from 0.01 to 1
      #         else:  # Second half
      #             scale = 1.0
              
      #         # Generate noise with scaled variance
      #         # scaled_std = u_std_d[i] * scale
      #         scaled_std = u_std_d[i] * 1
      #         noise_samples_d[block_id, t, i] = scaled_std * xoroshiro128p_normal_float32(rng_states, abs_thread_id)

      denom = 20
      for t in range(num_timesteps):
        for i in range(num_controls):
            # Linearly scaled standard deviation
            # scale = 1.0 - ((t / num_timesteps * (denom - 1)) / denom)
            scale = 1.0
            scaled_std = u_std_d[i] * scale

            # Generate noise with scaled variance
            noise_samples_d[block_id, t, i] = scaled_std * xoroshiro128p_normal_float32(rng_states, abs_thread_id)

  @staticmethod
  @cuda.jit(fastmath=True)
  def sample_noise_ou_numba(rng_states, theta, mu, sigma, dt, noise_samples_d):
      bid = cuda.blockIdx.x
      tid = cuda.threadIdx.x
      num_controls = noise_samples_d.shape[2]
      abs_tid = bid * noise_samples_d.shape[1] + tid

      for i in range(num_controls):
          # Initialize the noise value
          if tid == 0:
              prev_noise = mu
          else:
              prev_noise = noise_samples_d[bid, tid - 1, i]
          
          # Generate OU noise
          dx = theta * (mu - prev_noise) * dt + sigma[i] * math.sqrt(dt) * xoroshiro128p_normal_float32(rng_states, abs_tid)
          noise_samples_d[bid, tid, i] = prev_noise + dx

# Kernel wrapper that calls dynamics_update on a single state/control pair.
@cuda.jit
def dynamics_update_wrapper(x_in, u, dt, contact_normal, inertia_mass, plane, x_out, cf_out):
    # x_in: device array of shape (12,)
    # u: device array of shape (6,)
    # dt: float32
    # contact_normal: device array of shape (3,)
    # inertia_mass: device array of shape (4,)
    # x_out: device array of shape (12,)
    local_x = cuda.local.array(12, float32)
    
    for i in range(12):
        local_x[i] = x_in[i]
    
    # dynamics_update_contact(local_x, u, dt, contact_normal, inertia_mass, cf_out)
    # dynamics_update(local_x, u, dt, contact_normal, inertia_mass, cf_out)
    dynamics_update_lcp_contact_force(local_x, u, dt, contact_normal, inertia_mass, plane, cf_out)
    for i in range(12):
        x_out[i] = local_x[i]

# Helper function to simulate one step using the GPU kernel wrapper.
def simulate_dynamics(x, u, dt, contact_normal, inertia_mass, plane):
    # Ensure proper types: convert to np.float32
    x_in = np.array(x, dtype=np.float32)
    u = np.array(u, dtype=np.float32)
    dt = np.float32(dt)
    contact_normal = np.array(contact_normal, dtype=np.float32)
    inertia_mass = np.array(inertia_mass, dtype=np.float32)
    x_in_d = cuda.to_device(x_in)
    u_d = cuda.to_device(u)
    x_out_d = cuda.device_array((12,), dtype=np.float32)
    cf_out_d = cuda.device_array((3,), dtype=np.float32)
    dynamics_update_wrapper[1, 1](x_in_d, u_d, dt, contact_normal, inertia_mass, plane, x_out_d, cf_out_d)
    return (x_out_d.copy_to_host(), cf_out_d.copy_to_host())

def dynamics_update_euler(state, control_inputs, cf, dt, mppi_params):
  """
  Single-step Euler integration with contact impulses applied exactly once per dt.
  We simply call simulate_dynamics() once, which internally:
    - calls the GPU kernel
    - applies the contact impulse if needed
    - updates the state for the entire dt
  """
  # simulate_dynamics() is your GPU wrapper that calls dynamics_update_lcp_contact_force once
  next_state, next_cf = simulate_dynamics(
      state,
      control_inputs,
      dt,
      CONTACT_NORMAL,
      mppi_params['inertia_mass'],
      mppi_params['plane']
  )
  return (next_state, next_cf)

if __name__ == "__main__":
    num_controls = 6
    num_states = 12
    cfg = Config(
            T=2,                # Horizon length in seconds
            dt=0.3,        # Time step
            num_control_rollouts=1024*4,
            num_controls=6,
            num_states=12,
            num_vis_state_rollouts=1,
            seed=1
        )
    x0 = np.array([-0.5,0, 0, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    # x0 = np.array([-1.5,0, 0, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([2,-1, 3, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    # xgoal = np.array([2,-1, 3, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([0,0, 0.8, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([0.2,-0.2, 0.8, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    xgoal = np.array([0.2,-1, 0.8, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    fgoal = np.array([5, 0, 0])

    
    mppi_params = {
            'dt': cfg.dt,
            'x0': x0,
            'xgoal': xgoal,
            'fgoal': fgoal,
            'plane': np.array([1, 0, 0, -1.3]),
            'goal_tolerance': 0.001,
            'dist_weight': 2000,
            'lambda_weight': 10,
            'num_opt': 8,
            'u_std': np.array([0.5, 0.5, 0.5, 0.001, 0.001, 0.001]),
            'vrange': np.array([-10.0, 10.0]),
            'wrange': np.array([-0.1, 0.1]),
            'weights': np.array([
                19550, 19550, 24840,
                1, 1, 1,
                25500, 255000, 25500,
                1, 1, 1,
                1, 100, 1, 100, 200,
                100, 100, 100
            ]),
            "inertia_mass": np.array([0.21, 0.21, 0.4, 6.15])
        }

    mppi_controller = MPPI_Numba(cfg)
    mppi_controller.set_params(mppi_params)

    max_steps = 500

    
    # Loop
    
    xhist = np.zeros((max_steps+1, num_states))*np.nan
    fhist = np.zeros((max_steps+1, 3))*np.nan
    uhist = np.zeros((max_steps, num_controls))*np.nan
    xhist[0] = x0
    fhist[0] = np.zeros(3)


    for t in range(max_steps):
        # Solve
        useq = mppi_controller.solve()
        u_curr = useq[0]
        phi, theta, psi = xhist[t, 6:9]
        gravity_vector_world = np.array([0, 0, 9.81*7.00])
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                       np.cos(phi)*np.cos(theta)]
        ])
        gravity_body = np.dot(R.T, gravity_vector_world)  # Rotate gravity to body frame

        uhist[t] = u_curr.copy()

        # Simulate state forward 
        xhist[t+1, :], fhist[t+1, :] = dynamics_update_euler(xhist[t, :], u_curr, fhist[t, :], cfg.dt, mppi_params)

        print(t)
        # Update MPPI state (x0, useq)
        mppi_controller.shift_and_update(xhist[t+1], useq, num_shifts=1)

    # Assuming xgoal is your goal position and it has appropriate values for each state
    x_goal, y_goal, z_goal = xgoal[:3]
    roll_goal, pitch_goal, yaw_goal = xgoal[6:9]

    fig, axs = plt.subplots(5, 3, figsize=(12, 9))

    goals = np.concatenate([xgoal[:3], xgoal[6:9]])
    labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    for i, (ax, idx) in enumerate(zip(axs.flat[:6], [0, 1, 2, 6, 7, 8])):
        ax.plot(xhist[:, idx] * (180 / np.pi if i >= 3 else 1), label=labels[i])
        goal = goals[i] * (180 / np.pi if i >= 3 else 1)
        ax.axhline(goal, ls="--", color="green", label="goal")
        ax.set_ylabel("deg" if i >= 3 else "m")
        ax.set_xlabel("step")
        ax.legend()

    for i, (ax, idx) in enumerate(zip(axs.flat[6:12], range(6))):
        ax.plot(uhist[:, idx])
        ax.set_ylabel("N" if idx < 3 else "Nm")
        ax.set_xlabel("step")
        ax.set_title(f"u[{idx}]")

    for i, (ax, idx) in enumerate(zip(axs.flat[12:], range(3))):
        ax.plot(fhist[:, idx])
        ax.set_ylabel("N")
        ax.set_xlabel("step")
        ax.set_title(f"F_contact[{idx}]")

    
    fig2 = plt.figure()

    plt.plot(uhist[:, 0])

    plt.tight_layout()
    plt.show()
