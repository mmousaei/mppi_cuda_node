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
max_rec_blocks = rec_max_control_rollouts = int(1e6) # Though theoretically limited by max_blocks on GPU
rec_min_control_rollouts = 100

CONTACT_NORMAL = np.array([-1, 0, 0], dtype=np.float32)
# CONTACT_NORMAL = cuda.to_device(CONTACT_NORMAL_numpy)
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
# Define stage and terminal cost weights for each state dimension
STAGE_COST_WEIGHTS = np.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0], dtype=np.float32)  # Example weights
TERMINAL_COST_WEIGHTS = np.array([1000, 1000, 2000, 0, 0, 0, 5000, 5000, 10000, 0, 0, 0], dtype=np.float32)  # Example weights



# Stage costs (device function)
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def stage_cost(dist2, dist_weight):
  return dist_weight*dist2 # squared term makes the robot move faster

# Terminal costs (device function)
@cuda.jit('float32(float32, boolean)', device=True, inline=True)
def term_cost(dist2, goal_reached):
  return (1-np.float32(goal_reached))*dist2


@cuda.jit(device=True, fastmath=True)
def dynamics_update_lcp_contact_force(x, u, dt, contact_normal, inertia_mass,
                                      f_contact_out=None):
    """
    Single-plane LCP approach, ignoring friction & rotation from contact.
    Also computes contact force = normal impulse / dt if end-effector is penetrating.
    
    x: state array [pos(3), vel(3), rpy(3), ang_vel(3)] in float32
    u: control [Fx_body, Fy_body, Fz_body, Mx, My, Mz] in float32
    dt: time step
    inertia_mass: [Ixx, Iyy, Izz, mass]
    plane_x: the boundary in the world X dimension
    arm_length: distance from UAV origin to end-effector in the UAV +X_b direction
    f_contact_out: optional array of shape (3,) to store the contact force in world frame
                   If None, no output is stored.
    """
    plane_x=1.3  # contact plane at x = plane_x
    arm_length=1.2
    # ------------------ 1) FREE-FLIGHT INTEGRATION ------------------
    I_xx = inertia_mass[0]
    I_yy = inertia_mass[1]
    I_zz = inertia_mass[2]
    mass = inertia_mass[3]
    g = 9.81

    # Current state
    px, py, pz = x[0], x[1], x[2]      # position (world)
    vx, vy, vz = x[3], x[4], x[5]      # velocity (world)
    phi, theta, psi = x[6], x[7], x[8] # roll, pitch, yaw
    wx, wy, wz = x[9], x[10], x[11]    # angular velocity (body frame)

    # Rotation from body to world (Z-Y-X):
    sin_phi  = math.sin(phi)
    cos_phi  = math.cos(phi)
    sin_theta= math.sin(theta)
    cos_theta= math.cos(theta)
    sin_psi  = math.sin(psi)
    cos_psi  = math.cos(psi)

    R00 = cos_theta*cos_psi
    R01 = cos_theta*sin_psi
    R02 = -sin_theta
    R10 = sin_phi*sin_theta*cos_psi - cos_phi*sin_psi
    R11 = sin_phi*sin_theta*sin_psi + cos_phi*cos_psi
    R12 = sin_phi*cos_theta
    R20 = cos_phi*sin_theta*cos_psi + sin_phi*sin_psi
    R21 = cos_phi*sin_theta*sin_psi - sin_phi*cos_psi
    R22 = cos_phi*cos_theta

    # Body-frame forces/torques:
    Fx_b, Fy_b, Fz_b = u[0], u[1], u[2]
    Mx_b, My_b, Mz_b = u[3], u[4], u[5]

    # Convert body forces to world frame
    Fx_w = R00*Fx_b + R01*Fy_b + R02*Fz_b
    Fy_w = R10*Fx_b + R11*Fy_b + R12*Fz_b
    Fz_w = R20*Fx_b + R21*Fy_b + R22*Fz_b

    # Gravity
    Fx_w_total = Fx_w
    Fy_w_total = Fy_w
    Fz_w_total = Fz_w - mass*g

    # Euler angle derivatives
    phi_dot   = wx + sin_phi*math.tan(theta)*wy + cos_phi*math.tan(theta)*wz
    theta_dot = cos_phi*wy - sin_phi*wz
    psi_dot   = (sin_phi*wy + cos_phi*wz)/cos_theta

    # Angular accelerations (body frame)
    wx_dot = (Mx_b + (I_yy - I_zz)*wy*wz) / I_xx
    wy_dot = (My_b + (I_zz - I_xx)*wx*wz) / I_yy
    wz_dot = (Mz_b + (I_xx - I_yy)*wx*wy) / I_zz

    # Integrate forward (simple Euler)
    px_new   = px + dt*vx
    py_new   = py + dt*vy
    pz_new   = pz + dt*vz
    vx_new   = vx + dt*(Fx_w_total / mass)
    vy_new   = vy + dt*(Fy_w_total / mass)
    vz_new   = vz + dt*(Fz_w_total / mass)
    phi_new   = phi   + dt*phi_dot
    theta_new = theta + dt*theta_dot
    psi_new   = psi   + dt*psi_dot
    wx_new = wx + dt*wx_dot
    wy_new = wy + dt*wy_dot
    wz_new = wz + dt*wz_dot

    # Store them (temp)
    x[0], x[1], x[2] = px_new, py_new, pz_new
    x[3], x[4], x[5] = vx_new, vy_new, vz_new
    x[6], x[7], x[8] = phi_new, theta_new, psi_new
    x[9], x[10], x[11] = wx_new, wy_new, wz_new

    # Initialize contact force to zero
    if f_contact_out is not None:
        f_contact_out[0] = 0.0
        f_contact_out[1] = 0.0
        f_contact_out[2] = 0.0

    # ------------------ 2) LCP CONTACT CORRECTION --------------------
    # Recompute rotation with updated angles
    phi2, theta2, psi2 = x[6], x[7], x[8]
    sphi2, cphi2 = math.sin(phi2), math.cos(phi2)
    stheta2, ctheta2 = math.sin(theta2), math.cos(theta2)
    spsi2, cpsi2 = math.sin(psi2), math.cos(psi2)

    RR00 = ctheta2*cpsi2
    RR01 = ctheta2*spsi2
    RR02 = -stheta2
    RR10 = sphi2*stheta2*cpsi2 - cphi2*spsi2
    RR11 = sphi2*stheta2*spsi2 + cphi2*cpsi2
    RR12 = sphi2*ctheta2
    RR20 = cphi2*stheta2*cpsi2 + sphi2*spsi2
    RR21 = cphi2*stheta2*spsi2 - sphi2*cpsi2
    RR22 = cphi2*ctheta2

    px2, py2, pz2 = x[0], x[1], x[2]
    vx2, vy2, vz2 = x[3], x[4], x[5]

    # End-effector in world
    ee_wx = px2 + (RR00*arm_length)
    ee_wy = py2 + (RR10*arm_length)
    ee_wz = pz2 + (RR20*arm_length)

    # If end-effector is beyond plane_x => clamp position & zero normal velocity
    if ee_wx > plane_x:
        # (A) shift so E.E. is on plane
        penetration = ee_wx - plane_x
        # We'll do a naive shift in x
        x[0] -= penetration  # shift body in negative x


        # (B) find E.E. velocity in world X.  (We are ignoring angular velocity for brevity.)
        # If that velocity is positive => we apply impulse
        if vx2 > 0.0:
            # J = - mass * vx2   => it zeroes out vx
            # F_contact_x = J / dt
            J = - mass * vx2
            Fx_contact = J / dt  # This is the contact force in +X

            # apply that impulse => new vx
            x[3] = 0.0

            # store it if requested
            if f_contact_out is not None:
                f_contact_out[0] = Fx_contact
                f_contact_out[1] = 0.0
                f_contact_out[2] = 0.0

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
    self.dz = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
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
    return vrange_d, wrange_d, xgoal_d, \
           goal_tolerance_d, lambda_weight_d, \
           u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d, \
           cost_weights_d, inertia_mass_d


  def solve_with_nominal_dynamics(self):
    """
    Launch GPU kernels that use nominal dynamics but adjsuts cost function based on worst-case linear speed.
    """
    
    vrange_d, wrange_d, xgoal_d, goal_tolerance_d, lambda_weight_d, \
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
    # Calculate the gravity vector in body frame
    # gravity_vector = np.zeros((self.num_controls), dtype=np.float32)
    # gravity_vector[0] = - 9.81 * (np.cos(new_x0[6]) * np.sin(new_x0[7]) * np.cos(new_x0[8]) + np.sin(new_x0[6]) * np.sin(new_x0[8]))
    # gravity_vector[1] = - 9.81 * (np.cos(new_x0[6]) * np.sin(new_x0[7]) * np.sin(new_x0[8]) - np.sin(new_x0[6]) * np.cos(new_x0[8]))
    # gravity_vector[2] = - 9.81 * (np.cos(new_x0[6]) * np.cos(new_x0[7]))
    # u_cur[:3] += gravity_vector
    # print("gravity  =  ", gravity_vector)
    # self.u_seq0 = gravity_vector
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
      # dynamics_update(x_curr, u_noisy, dt_d, CONTACT_NORMAL, inertia_mass_d, cf)
      dynamics_update_lcp_contact_force(x_curr, u_noisy, dt_d, CONTACT_NORMAL, inertia_mass_d, cf)

      w_pose_xy = 4500
      w_pose_z =  5300
      w_vel = 150
      w_att = 75000
      w_omega = 500
      w_cont = 1
      w_cont_m = 1
      w_cont_f = 1
      w_cont_M = 1
      w_term = 500

      w_control_rate_fx = 0
      w_control_rate_fy = 0
      w_control_rate_fz = 0
      w_control_rate_mx = 0
      w_control_rate_my = 0
      w_control_rate_mz = 0

      # If else statements will be expensive
      dist_to_goal2 = cost_weights_d[0]*((xgoal_d[0]-x_curr[0])**2) + cost_weights_d[1]*((xgoal_d[1]-x_curr[1])**2) + cost_weights_d[2]*((xgoal_d[2]-x_curr[2])**2) \
                    + cost_weights_d[3]*((xgoal_d[3]-x_curr[3])**2) + cost_weights_d[4]*((xgoal_d[4]-x_curr[4])**2) + cost_weights_d[5]*((xgoal_d[5]-x_curr[5])**2)\
                    + cost_weights_d[6]*((xgoal_d[6]-x_curr[6])**2) + cost_weights_d[7]*((xgoal_d[7]-x_curr[7])**2) + cost_weights_d[8]*((xgoal_d[8]-x_curr[8])**2)\
                    + cost_weights_d[9]*((xgoal_d[9]-x_curr[9])**2) + cost_weights_d[10]*((xgoal_d[10]-x_curr[10])**2) + cost_weights_d[11]*(xgoal_d[11]-x_curr[11])**2\
                    + cost_weights_d[12]*(50*(u_nom[0]**2) + (u_nom[1]**2) + ((u_nom[2] - inertia_mass_d[3]*9.81)**2))\
                    + cost_weights_d[13]*((u_nom[3]**2) + (u_nom[4]**2) + (u_nom[5]**2)) + 100 * (cf[0] + 5) ** 2 
                    
      costs_d[bid]+= stage_cost(dist_to_goal2, dist_weight_d)

    # Add obstacle costs
      # num_obs = len(obs_pos_d)
      # for obs_i in range(num_obs):
      #   op = obs_pos_d[obs_i]
      #   dist_diff = (x_curr[0]-op[0])**2+(x_curr[1]-op[1])**2-obs_r_d[obs_i]**2
      #   costs_d[bid] += (1-numba.float32(dist_diff>0))*obs_cost_d

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

  # @staticmethod
  # @cuda.jit(fastmath=True)
  # def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
  #   """
  #   Should be invoked as sample_noise_numba[NUM_U_SAMPLES, NUM_THREADS].
  #   noise_samples_d.shape is assumed to be (num_rollouts, time_steps, 2)
  #   Assume each thread corresponds to one time step
  #   For consistency, each block samples a sequence, and threads (not too many) work together over num_steps.
  #   This will not work if time steps are more than max_threads_per_block (usually 1024)
  #   """
    
  #   block_id = cuda.blockIdx.x
  #   thread_id = cuda.threadIdx.x
  #   abs_thread_id = cuda.grid(1)

  #   noise_samples_d[block_id, thread_id, 0] = u_std_d[0]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
  #   noise_samples_d[block_id, thread_id, 1] = u_std_d[1]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
  #   noise_samples_d[block_id, thread_id, 2] = u_std_d[2]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
  #   noise_samples_d[block_id, thread_id, 3] = u_std_d[3]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
  #   noise_samples_d[block_id, thread_id, 4] = u_std_d[4]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
  #   noise_samples_d[block_id, thread_id, 5] = u_std_d[5]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)


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
  
if __name__ == "__main__":
    num_controls = 6
    num_states = 12
    cfg = Config(
            T=2,                # Horizon length in seconds
            dt=0.3,        # Time step
            num_control_rollouts=1024*16,
            num_controls=6,
            num_states=12,
            num_vis_state_rollouts=1,
            seed=1
        )
    # x0 = np.array([-0.5,0, 0, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    x0 = np.array([-1.5,0, 0, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([2,-1, 3, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    # xgoal = np.array([2,-1, 3, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([0,0, 0.8, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([0.2,-0.2, 0.8, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    xgoal = np.array([0.4,-1, 0.8, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
  

    
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
            "inertia_mass": np.array([0.21, 0.21, 0.4, 6.15])
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
    
    mpc = MPC(mpc_params)
    
    def hex_dynamics(x, u, mppi_params):
        p, v, Psi, omega = np.split(x, 4)
        f_T, m_T = u[:3], u[3:]
        phi, theta, psi = Psi
        J = np.diag(mppi_params['inertia_mass'][:3])
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]
        ])
        gravity_world = np.array([0, 0, -9.81])
        gravity_body = np.dot(R.T, gravity_world)  # Rotate gravity to body frame

        nu = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])

        p_dot = v
        v_dot = (1/mppi_params['inertia_mass'][3]) * f_T + gravity_body
        psi_dot = np.dot(nu, omega)
        omega_dot = np.dot(np.linalg.inv(J), m_T - np.cross(omega, np.dot(J, omega)))

        return np.concatenate([p_dot, v_dot, psi_dot, omega_dot])
    def forward_simulate_for_mpc_target(optimal_control_seq, current_state, mppi_params):
        """
        Forward simulate using the first MPPI control (for one time step)
        to obtain a target state that MPC can track.
        """
        mppi_u = optimal_control_seq[0, :].copy()

        # (Optional) Gravity compensation could be applied here if desired.
        # Forward-simulate using a simple RK4 integration:
        next_state = dynamics_update_rk4(current_state.copy(), mppi_u, mppi_params['dt'], mppi_params)
        # Zero-out the angular velocity components for the target
        # next_state[6:] = np.zeros(6)

        # next_state_filtered = self.lpf.filter(next_state.copy())
        # next_state_filtered[6:9] = np.clip(next_state_filtered[6:9], -0.1, 0.1)
        return next_state

    # Kernel wrapper that calls dynamics_update on a single state/control pair.
    @cuda.jit
    def dynamics_update_wrapper(x_in, u, dt, contact_normal, inertia_mass, x_out, cf_out):
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
        dynamics_update_lcp_contact_force(local_x, u, dt, contact_normal, inertia_mass, cf_out)
        for i in range(12):
            x_out[i] = local_x[i]

    # Helper function to simulate one step using the GPU kernel wrapper.
    def simulate_dynamics(x, u, dt, contact_normal, inertia_mass):
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
        dynamics_update_wrapper[1, 1](x_in_d, u_d, dt, contact_normal, inertia_mass, x_out_d, cf_out_d)
        return (x_out_d.copy_to_host(), cf_out_d.copy_to_host())

    def dynamics_update_rk4(state, control_inputs, cf, dt, mppi_params):
      """
      RK4 integration using the GPU kernel-based dynamics_update.
      
      This function first converts the inputs to np.float32, then defines a derivative function f(x)
      that uses simulate_dynamics (which wraps the dynamics_update kernel) to compute an approximation of dx/dt.
      Finally, the RK4 formulation is used.
      
      Note: The simulate_dynamics helper (defined elsewhere) calls a kernel wrapper that properly handles the
      dynamics_update device function and data type conversions.
      """
      # Ensure all arrays and dt are np.float32
      state = np.array(state, dtype=np.float32)
      control_inputs = np.array(control_inputs, dtype=np.float32)
      dt = np.float32(dt)
      inertia_mass = np.array(mppi_params['inertia_mass'], dtype=np.float32)
      contact_normal = CONTACT_NORMAL.astype(np.float32)

      # Helper function to approximate the time derivative using the GPU-based dynamics update.
      def f(x, cf):
          # simulate_dynamics calls our kernel wrapper (which internally uses dynamics_update)
          x_next, cf_next = simulate_dynamics(x, control_inputs, dt, contact_normal, inertia_mass)
          return ((x_next - x) / dt, (cf_next - cf) / dt)

      # Compute RK4 increments
      k1, kf1 = f(state, cf)
      k2, kf2 = f(state + k1 / 2, cf + kf1 / 2)
      k3, kf3 = f(state + k2 / 2, cf + kf2 / 2)
      k4, kf4 = f(state + k3, cf + kf3)

      # k1 = hex_dynamics(state, control_inputs, mppi_params) * dt
      # k2 = hex_dynamics(state + k1 / 2, control_inputs, mppi_params) * dt 
      # k3 = hex_dynamics(state + k2 / 2, control_inputs, mppi_params) * dt
      # k4 = hex_dynamics(state + k3, control_inputs, mppi_params) * dt

      next_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
      next_cf = cf + dt * (kf1 + 2 * kf2 + 2 * kf3 + kf4) / 6
      return (next_state, next_cf)
    
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
          mppi_params['inertia_mass']
      )
      return (next_state, next_cf)
    # Loop
    
    xhist = np.zeros((max_steps+1, num_states))*np.nan
    fhist = np.zeros((max_steps+1, 3))*np.nan
    uhist = np.zeros((max_steps, num_controls))*np.nan
    mpctargethist = np.zeros((max_steps+1, num_states))*np.nan
    xhist[0] = x0
    fhist[0] = np.zeros(3)
    mpctargethist[0] = x0

    vis_xlim = [-1, 8]
    vis_ylim = [-1, 6]
    mpc_target  = np.zeros(12)
    plot_every_n = 15
    
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
        if t % 10 == 0:
          # mpc_target = forward_simulate_for_mpc_target(useq, xhist[t, :], mppi_params)
          mpc_target = np.zeros(12)
        # u_mpc = mpc.compute_control(xhist[t, :], mpc_target, np.zeros(6), mpc_params['dt'])
        u_mpc = np.zeros(6)
        mpctargethist[t+1, :] = mpc_target.copy()
        # u_curr[:3] += gravity_body
        if use_mpc:
          uhist[t] = u_mpc
        else:
          uhist[t] = u_curr.copy()

        # Simulate state forward 
        if use_mpc:
          # xhist[t+1, :] = dynamics_update_sim(xhist[t, :], u_mpc, mpc_params['dt'])
          xhist[t+1, :], fhist[t+1, :] = dynamics_update_rk4(xhist[t, :], u_mpc, fhist[t, :], mpc_params['dt'], mppi_params)
        else:
          # xhist[t+1, :] = dynamics_update_sim(xhist[t, :], u_curr, cfg.dt)
          # xhist[t+1, :], fhist[t+1, :] = dynamics_update_rk4(xhist[t, :], u_curr, fhist[t, :], cfg.dt, mppi_params)
          xhist[t+1, :], fhist[t+1, :] = dynamics_update_euler(xhist[t, :], u_curr, fhist[t, :], cfg.dt, mppi_params)
        # print("x: ", xhist[t+1, :])
        print(t)
        # Update MPPI state (x0, useq)
        mppi_controller.shift_and_update(xhist[t+1], useq, num_shifts=1)

    # Assuming xgoal is your goal position and it has appropriate values for each state
    x_goal, y_goal, z_goal = xgoal[:3]
    roll_goal, pitch_goal, yaw_goal = xgoal[6:9]

    fig, axs = plt.subplots(5, 3, figsize=(12, 9))  # Create 3 subplots, one for each series

    # Plot X with Goal
    axs[0][0].plot(xhist[:, 0], label='x')
    axs[0][0].axhline(x_goal, color='green', linestyle='--', label='X Goal')  # X Goal
    axs[0][0].set_title('X')
    axs[0][0].set_xlabel('Time Steps')
    axs[0][0].set_ylabel('m')
    axs[0][0].legend()

    # Plot Y with Goal
    axs[0][1].plot(xhist[:, 1], label='y')
    axs[0][1].axhline(y_goal, color='green', linestyle='--', label='Y Goal')  # Y Goal
    axs[0][1].set_title('Y')
    axs[0][1].set_xlabel('Time Steps')
    axs[0][1].set_ylabel('m')
    axs[0][1].legend()

    # Plot Z with Goal
    axs[0][2].plot(xhist[:, 2], label='z')
    axs[0][2].axhline(z_goal, color='green', linestyle='--', label='Z Goal')  # Z Goal
    axs[0][2].set_title('Z')
    axs[0][2].set_xlabel('Time Steps')
    axs[0][2].set_ylabel('m')
    axs[0][2].legend()

    # Plot Roll with Goal
    axs[1][0].plot(xhist[:, 6]*180/np.pi, label='roll')
    axs[1][0].axhline(roll_goal*180/np.pi, color='green', linestyle='--', label='Roll Goal')  # Roll Goal
    axs[1][0].set_title('Roll')
    axs[1][0].set_xlabel('Time Steps')
    axs[1][0].set_ylabel('Angle (degrees)')
    axs[1][0].legend()

    # Plot Pitch with Goal
    axs[1][1].plot(xhist[:, 7]*180/np.pi, label='pitch')
    axs[1][1].axhline(pitch_goal*180/np.pi, color='green', linestyle='--', label='Pitch Goal')  # Pitch Goal
    axs[1][1].set_title('Pitch')
    axs[1][1].set_xlabel('Time Steps')
    axs[1][1].set_ylabel('Angle (degrees)')
    axs[1][1].legend()

    # Plot Yaw with Goal
    axs[1][2].plot(xhist[:, 8]*180/np.pi, label='yaw')
    axs[1][2].axhline(yaw_goal*180/np.pi, color='green', linestyle='--', label='Yaw Goal')  # Yaw Goal
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

    # # EE FORCE
    axs[4][0].plot(fhist[:, 0], label='X')
    axs[4][0].set_title('FX')
    axs[4][0].set_xlabel('Time Steps')
    axs[4][0].set_ylabel('m')
    axs[4][0].legend()
    # mpc target y
    axs[4][1].plot(fhist[:, 1], label='Y')
    axs[4][1].set_title('FY')
    axs[4][1].set_xlabel('Time Steps')
    axs[4][1].set_ylabel('m')
    axs[4][1].legend()
    # mpc target z
    axs[4][2].plot(fhist[:, 2], label='Z')
    axs[4][2].set_title('FZ')
    axs[4][2].set_xlabel('Time Steps')
    axs[4][2].set_ylabel('m')
    axs[4][2].legend()


    plt.tight_layout()  # Adjusts the subplots to fit in the figure area
    plt.show()
