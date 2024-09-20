import numpy as np
import math
import copy
import numba
import time
from numba import cuda, float32, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import matplotlib.pyplot as plt


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
               T=1, # Horizon (s)
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

DEFAULT_OBS_COST = 1e3
DEFAULT_DIST_WEIGHT = 10
# Define stage and terminal cost weights for each state dimension
STAGE_COST_WEIGHTS = np.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0], dtype=np.float32)  # Example weights
TERMINAL_COST_WEIGHTS = np.array([1000, 1000, 2000, 0, 0, 0, 5000, 5000, 10000, 0, 0, 0], dtype=np.float32)  # Example weights


def dynamics_update_sim(x, u, dt):
  # The dynamics update for hexarotor
  # I_xx = 0.23038337
  # I_yy = 0.11771596
  # I_zz = 0.11392979
  I_xx = 0.115125971
  I_yy = 0.116524229
  I_zz = 0.230387752

  mass = 2.302499999999999
  g = 9.81

  
  x_next = x.copy()

  x_next[0] += dt * x[3] 
  x_next[1] += dt * x[4]
  x_next[2] += dt * x[5]
  
  x_next[3] += dt * ((1/mass) * u[0] + g * np.sin(x[7]))
  x_next[4] += dt * ((1/mass) * u[1] - g * np.cos(x[7]) * np.sin(x[6]))
  x_next[5] += dt * ((1/mass) * u[2] - g * np.cos(x[7]) * np.cos(x[6]))

  x_next[6] += dt*(x[9] + x[10]*(math.sin(x[6])*math.tan(x[7])) + x[11]*(math.cos(x[6])*math.tan(x[7])))
  x_next[7] += dt*( x[10]*math.cos(x[6]) - x[11]*math.sin(x[6]))
  x_next[8] += dt*( x[10]*math.sin(x[6])/math.cos(x[7]) + x[11]*math.cos(x[6])/math.cos(x[7]))
  # x_next[6] += dt * ( x[9]*math.cos(x[8])*math.cos(x[7]) + x[10]*(math.sin(x[6])*math.sin(x[7])*math.cos(x[8]) - math.sin(x[8])*math.cos(x[6])) + x[11]*(math.sin(x[6])*math.sin(x[8]) + math.sin(x[7])*math.cos(x[6])*math.cos(x[8])) )
  # x_next[7] += dt * ( x[9]*math.sin(x[8])*math.cos(x[7]) + x[10]*(math.sin(x[6])*math.sin(x[8])*math.sin(x[7]) + math.cos(x[6])*math.cos(x[8])) + x[11]*(-math.sin(x[6])*math.cos(x[8]) + math.sin(x[8])*math.sin(x[7])*math.cos(x[6])) )
  # x_next[8] += dt * ( -x[9]*math.sin(x[7]) + x[10]*math.sin(x[6])*math.cos(x[7]) + x[11]*math.cos(x[6])*math.cos(x[7]) )

  x_next[9]  += dt*((1/I_xx) * (u[3] + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
  x_next[10] += dt*((1/I_yy) * (u[4] - I_xx * x[9] *  x[11] + I_zz * x[9] *  x[11]))
  x_next[11] += dt*((1/I_zz) * (u[5] + I_xx * x[9] *  x[10] - I_yy * x[9] *  x[10]))

  return x_next

# Stage costs (device function)
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def stage_cost(dist2, dist_weight):
  return dist_weight*dist2 # squared term makes the robot move faster

# Terminal costs (device function)
@cuda.jit('float32(float32, boolean)', device=True, inline=True)
def term_cost(dist2, goal_reached):
  return (1-np.float32(goal_reached))*dist2*10000


@cuda.jit(device=True, fastmath=True)
def calculate_contact_force_moment_naiive(x, u, A, B, C, D, ABC_sq, contact_normal_sq, contact_normal):
  
  arm_length = 1.2
  contact_threshold = 0.01
  ee_pose_x = x[0] + arm_length*math.cos(x[7])*math.cos(x[8])
  ee_pose_y = x[1] + arm_length*math.cos(x[7])*math.sin(x[8])
  ee_pose_z = x[2] + arm_length*math.sin(x[7])

  ABC_sq = math.sqrt(A**2 + B**2 + C**2)
  dist_from_contact_plane = ((A * ee_pose_x + B * ee_pose_y + C * ee_pose_z + D) / ABC_sq)
  force_dot = u[0] * contact_normal[0] + u[1] * contact_normal[1] * u[2] * contact_normal[2]
  contact_bitmask = dist_from_contact_plane < contact_threshold
  
  contact_force_x = (- force_dot / contact_normal_sq * contact_normal[0]) *  contact_bitmask
  contact_force_y = (- force_dot / contact_normal_sq * contact_normal[1]) *  contact_bitmask
  contact_force_z = (- force_dot / contact_normal_sq * contact_normal[2]) *  contact_bitmask

  velocity_dot = x[3] * contact_normal[0] + x[4] * contact_normal[1] + x[5] * contact_normal[2]

  contact_velocity_x = - velocity_dot / contact_normal_sq * contact_normal[0] * contact_bitmask
  contact_velocity_y = - velocity_dot / contact_normal_sq * contact_normal[1] * contact_bitmask
  contact_velocity_z = - velocity_dot / contact_normal_sq * contact_normal[2] * contact_bitmask

  contact_moment_x = -(math.cos(x[7]) * math.sin(x[8]) * arm_length * contact_force_z + math.sin(x[7]) * arm_length * contact_force_y)                  *  contact_bitmask
  contact_moment_y = -(-math.sin(x[7]) * arm_length * contact_force_x - math.cos(x[7]) * math.cos(x[8]) * arm_length * contact_force_z)                 *  contact_bitmask
  contact_moment_z = -(math.cos(x[7]) * math.cos(x[8]) * arm_length * contact_force_y - math.cos(x[7]) * math.sin(x[8]) * arm_length * contact_force_x) *  contact_bitmask

  return contact_force_x, contact_force_y, contact_force_z, contact_velocity_x, contact_velocity_y, contact_velocity_z, contact_moment_x, contact_moment_y, contact_moment_z


@cuda.jit(device=True, fastmath=True)
def dynamics_update(x, u, dt, contact_normal, inertia_mass):
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

  # contact_force_x, contact_force_y, contact_force_z, contact_velocity_x, contact_velocity_y, contact_velocity_z\
  #   , contact_moment_x, contact_moment_y, contact_moment_z = \
  #   calculate_contact_force_moment_naiive(x, u, A, B, C, D, ABC_sq, contact_normal_sq, contact_normal)
  
  contact_force_x, contact_force_y, contact_force_z, contact_velocity_x, contact_velocity_y, contact_velocity_z\
    , contact_moment_x, contact_moment_y, contact_moment_z = 0, 0, 0, 0, 0, 0, 0, 0, 0

  c = -300 # spring damping model

  fx_total = (u[0] + contact_force_x) - (c * (contact_velocity_x )) 
  fy_total = (u[1] + contact_force_y) - (c * (contact_velocity_y )) 
  fz_total = (u[2] + contact_force_z) - (c * (contact_velocity_z ))
  mx_total = u[3] + contact_moment_x 
  my_total = u[4] + contact_moment_y 
  mz_total = u[5] + contact_moment_z  

  x[0] += dt*x[3] 
  x[1] += dt*x[4]
  x[2] += dt*x[5]

  x[3] += dt*((1/mass) * fx_total)# + 9.81 * np.sin(x[7]))
  x[4] += dt*((1/mass) * fy_total)# - 9.81 * np.cos(x[7]) * np.sin(x[6]))
  x[5] += dt*((1/mass) * fz_total)# - 9.81 * np.cos(x[7]) * np.cos(x[6]))

  x[6] += dt*(x[9] + x[10]*(math.sin(x[6])*math.tan(x[7])) + x[11]*(math.cos(x[6])*math.tan(x[7])))
  x[7] += dt*( x[10]*math.cos(x[6]) - x[11]*math.sin(x[6]))
  x[8] += dt*( x[10]*math.sin(x[6])/math.cos(x[7]) + x[11]*math.cos(x[6])/math.cos(x[7]))
  
  x[9]  += dt*((1/I_xx) * (mx_total + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
  x[10] += dt*((1/I_yy) * (my_total - I_xx * x[9] *  x[11] + I_zz * x[9] *  x[11]))
  x[11] += dt*((1/I_zz) * (mz_total + I_xx * x[9] *  x[10] - I_yy * x[9] *  x[10]))

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
    self.use_ou = False
    self.theta = 1  # Example value
    self.mu = 0.0  # OU process mean
    self.sigma = 1  # Example value
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
    # mppi cost weights
    self.weights = None
    self.inertia_mass = None
    self.reset()

    
  def reset(self):
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, self.num_controls), dtype=np.float32)
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
    self.weights = params['weights']
    self.inertia_mass = params['inertia_mass']


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

    weights_d = cuda.to_device(self.weights.astype(np.float32))
    inertia_mass_d = cuda.to_device(self.inertia_mass.astype(np.float32))
    return vrange_d, wrange_d, xgoal_d, \
           goal_tolerance_d, lambda_weight_d, \
           u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d, \
           weights_d, inertia_mass_d

  def solve_with_nominal_dynamics(self):
    """
    Launch GPU kernels that use nominal dynamics but adjsuts cost function based on worst-case linear speed.
    """
    
    vrange_d, wrange_d, xgoal_d, goal_tolerance_d, lambda_weight_d, \
           u_std_d, x0_d, dt_d, obs_cost_d, obs_pos_d, obs_r_d, weights_d, inertia_mass_d = self.move_mppi_task_vars_to_device()
  
    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']
    # block_size = 1024  # Example block size, adjust based on your GPU's capability
    # grid_size = (self.num_control_rollouts + (block_size - 1)) // block_size
    # Example kernel launch configuration, adjust based on your specific needs
    threads_per_block = 256  # Or 512, depending on your GPU's capability
    total_threads = self.num_control_rollouts * self.num_controls
    blocks_per_grid = math.ceil(total_threads / threads_per_block)
    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      if self.use_ou:
        # Call OU noise sampling kernel
        self.sample_noise_ou_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, self.ou_alpha, self.sys_noise, self.ou_scale, self.d_ou_scale, self.num_steps,
            self.num_control_rollouts, self.num_controls, self.umin, self.umax, 
            self.last_controls_d, self.noise_samples_d, self.dz)

      else:
        self.sample_noise_numba[self.num_control_rollouts, self.num_steps](
            self.rng_states_d, u_std_d, self.noise_samples_d)
    #   self.sample_noise_numba[grid_size, block_size](
                #   self.rng_states_d, u_std_d, self.noise_samples_d)

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
        weights_d,
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
          obs_cost_d, 
          obs_pos_d, 
          obs_r_d,
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          x0_d, 
          dt_d,
          dist_weight_d,
          weights_d,
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
    u_diff = cuda.local.array(6, numba.float32)

    

    # printed=False
    for t in range(timesteps):
      # Nominal noisy control
      u_nom[0] = u_cur_d[t, 0] + noise_samples_d[bid, t, 0]
      u_nom[1] = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      u_nom[2] = u_cur_d[t, 2] + noise_samples_d[bid, t, 2]
      u_nom[3] = u_cur_d[t, 3] + noise_samples_d[bid, t, 3]
      u_nom[4] = u_cur_d[t, 4] + noise_samples_d[bid, t, 4]
      u_nom[5] = u_cur_d[t, 5] + noise_samples_d[bid, t, 5]

      u_diff[0] = u_nom[0] - u_cur_d[t-1, 0]
      u_diff[1] = u_nom[1] - u_cur_d[t-1, 1]
      u_diff[2] = u_nom[2] - u_cur_d[t-1, 2]
      u_diff[3] = u_nom[3] - u_cur_d[t-1, 3]
      u_diff[4] = u_nom[4] - u_cur_d[t-1, 4]
      u_diff[5] = u_nom[5] - u_cur_d[t-1, 5]

      # TODO: implement control limits  
      u_noisy = u_nom
      # u_noisy = max(vrange_d[0], min(vrange_d[1], v_nom))
      
      # Forward simulate
      dynamics_update(x_curr, u_noisy, dt_d, CONTACT_NORMAL, inertia_mass_d)

      # If else statements will be expensive
      dist_to_goal2 = weights_d[0]*((xgoal_d[0]-x_curr[0])**2) + weights_d[1]*((xgoal_d[1]-x_curr[1])**2) + weights_d[2]*((xgoal_d[2]-x_curr[2])**2) \
                    + weights_d[3]*((xgoal_d[3]-x_curr[3])**2 + (xgoal_d[4]-x_curr[4])**2 + (xgoal_d[5]-x_curr[5])**2)\
                    + weights_d[4]*((xgoal_d[6]-x_curr[6])**2) + weights_d[5]*((xgoal_d[7]-x_curr[7])**2) + weights_d[6]*((xgoal_d[8]-x_curr[8])**2)\
                    + weights_d[7]*((xgoal_d[9]-x_curr[9])**2 + (xgoal_d[10]-x_curr[10])**2 + (xgoal_d[11]-x_curr[11])**2)\
                    + weights_d[8]*((u_nom[0]**2) + (u_nom[1]**2) + (u_nom[2]**2))\
                    + weights_d[9]*((u_nom[3]**2) + (u_nom[4]**2) + (u_nom[5]**2))
                    
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
    costs_d[bid] += weights_d[12]*term_cost(dist_to_goal2, goal_reached)
    # Add Control cost 
    for t in range(timesteps):
      costs_d[bid] += weights_d[10]*lambda_weight_d*(
              (u_cur_d[t,0]/(u_std_d[0]**2))*noise_samples_d[bid, t,0] + (u_cur_d[t,1]/(u_std_d[1]**2))*noise_samples_d[bid, t, 1] + (u_cur_d[t,2]/(u_std_d[2]**2))*noise_samples_d[bid, t, 2]\
                 + weights_d[11]*((u_cur_d[t,3]/(u_std_d[3]**2))*noise_samples_d[bid, t, 3] + (u_cur_d[t,4]/(u_std_d[4]**2))*noise_samples_d[bid, t, 4] + (u_cur_d[t,5]/(u_std_d[5]**2))*noise_samples_d[bid, t, 5]))

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
    for ti in range(starti, endi):
      u_cur_d[ti, 0] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 0]))
      u_cur_d[ti, 1] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 1]))
      u_cur_d[ti, 2] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 2]))
      u_cur_d[ti, 3] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 3]))
      u_cur_d[ti, 4] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 4]))
      u_cur_d[ti, 5] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 5]))


  @staticmethod
  @cuda.jit(fastmath=True)
  def get_state_rollout_across_control_noise(
          state_rollout_batch_d, # where to store results
          x0_d, 
          dt_d,
          noise_samples_d,
          vrange_d,
          wrange_d,
          u_prev_d,
          u_cur_d,
          inertia_mass_d):
    """
    Do a fixed number of rollouts for visualization across blocks.
    Assume kernel is launched as get_state_rollout_across_control_noise[num_blocks, 1]
    The block with id 0 will always visualize the best control sequence. Other blocks will visualize random samples.
    """
    
    # Use block id
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    timesteps = len(u_cur_d)


    if bid==0:
      # Visualize the current best 
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = cuda.local.array(3, numba.float32)
      for i in range(3): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]
      
      for t in range(timesteps):
        # Nominal noisy control
        u_nom = u_cur_d[t, :]
        
        # Forward simulate
        dynamics_update(x_curr, u_nom, dt_d, CONTACT_NORMAL, inertia_mass_d)

        # Save state
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]
    else:
      
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = cuda.local.array(3, numba.float32)
      for i in range(3): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]

      
      for t in range(timesteps):
        # Nominal noisy control
        u_nom[0] = u_prev_d[t, 0] + noise_samples_d[bid, t, 0]
        u_nom[1] = u_prev_d[t, 1] + noise_samples_d[bid, t, 1]
        u_nom[2] = u_prev_d[t, 2] + noise_samples_d[bid, t, 2]
        u_nom[3] = u_prev_d[t, 3] + noise_samples_d[bid, t, 3]
        u_nom[4] = u_prev_d[t, 4] + noise_samples_d[bid, t, 4]
        u_nom[5] = u_prev_d[t, 5] + noise_samples_d[bid, t, 5]

        # TODO: implement control limits
        u_noisy = u_nom

        # # Nominal noisy control
        u_nom = u_prev_d[t, :]
        
        # Forward simulate
        dynamics_update(x_curr, u_noisy, dt_d, CONTACT_NORMAL, inertia_mass_d)

        # Save state
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]


  @staticmethod
  @cuda.jit(fastmath=True)
  def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
    """
    Should be invoked as sample_noise_numba[NUM_U_SAMPLES, NUM_THREADS].
    noise_samples_d.shape is assumed to be (num_rollouts, time_steps, 2)
    Assume each thread corresponds to one time step
    For consistency, each block samples a sequence, and threads (not too many) work together over num_steps.
    This will not work if time steps are more than max_threads_per_block (usually 1024)
    """
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    abs_thread_id = cuda.grid(1)

    noise_samples_d[block_id, thread_id, 0] = u_std_d[0]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 1] = u_std_d[1]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 2] = u_std_d[2]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 3] = u_std_d[3]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 4] = u_std_d[4]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 5] = u_std_d[5]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)

  @staticmethod
  @cuda.jit
  def sample_noise_ou_numba(rng_states, ou_alpha, sys_noise, ou_scale, d_ou_scale, T, K1, m, umin, umax, last_controls, control_noise, dz):
      tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
      ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

      if tx < K1 and ty < m:
          ou_alpha_f32 = float32(ou_alpha)
          ou_scale_f32 = float32(ou_scale)
          d_ou_scale_f32 = float32(d_ou_scale)
          T_f32 = float32(T)

          sys_noise_f32 = float32(sys_noise[ty])

          for t in range(T):
              if t == 0:
                  initial_noise = xoroshiro128p_normal_float32(rng_states, tx * m + ty) * sys_noise_f32 * (ou_scale_f32 / T_f32)
                  dz[tx, t, ty] = initial_noise
              else:
                  incremental_noise = xoroshiro128p_normal_float32(rng_states, tx * m + ty + K1 * t) * sys_noise_f32 * (d_ou_scale_f32 / T_f32)
                  dz[tx, t, ty] = ou_alpha_f32 * dz[tx, t - 1, ty] + (1 - ou_alpha_f32) * incremental_noise

              control_noise_val = last_controls[tx, t, ty] + dz[tx, t, ty]
              control_noise_val = max(umin[ty], min(control_noise_val, umax[ty]))
              control_noise[tx, t, ty] = control_noise_val
  
if __name__ == "__main__":
    num_controls = 6
    num_states = 12
    cfg = Config(T = 1,
            dt = 0.02,
            num_control_rollouts = 2048,#int(2e4), # Same as number of blocks, can be more than 1024
            num_controls = num_controls,
            num_states = num_states,
            num_vis_state_rollouts = 1,
            seed = 1)
    x0 = np.zeros(12)
    xgoal = np.array([1, -1, 2, 0, 0, 0, -0.3, 0.3, 0.3, 0, 0, 0])

    xyz = 150
    v = 15
    rpy = 1500
    omega = 100
    cont_f1 = 1
    cont_f2 = 5
    cont_m2 = 5
    cont_m1 = 1
    term = 100
    mppi_params = dict(
        # Task specification
        dt=cfg.dt,
        x0=x0, # Start state
        xgoal=xgoal, # Goal position

        # For risk-aware min time planning
        goal_tolerance=0.001,
        dist_weight=200, #  Weight for dist-to-goal cost.
        # dist_weights = np.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0]),
        lambda_weight=10.0, # Temperature param in MPPI
        num_opt=5, # Number of steps in each solve() function call.

        # Control and sample specification
        u_std=np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])*0.05, # Noise std for sampling linear and angular velocities.
        vrange = np.array([-10.0, 10.0]), # Linear velocity range.
        wrange=np.array([-0.5, 0.5]), # Angular velocity range.
        
        
        # dt = 0.02 tuning parameters
        # weights = np.array([150, 150, 300, 15, 1500, 1500, 3000, 100, 1, 5, 5, 1, 100]), # w_pose_x, w_pose_y, w_pose_z, w_vel, w_att_roll, w_att_pitch, w_att_yaw, w_omega, w_cont, w_cont_m, w_cont_f, w_cont_M, w_terminal
        weights = np.array([4550, 4550, 5300, 150, 75000, 35000, 85000, 500, 1, 5, 5, 1, 500]),
        inertia_mass = np.array([0.115125971, 0.116524229, 0.230387752, 2.302499999999999]) # I_xx, I_yy, I_zz, mass
    )

    mppi_controller = MPPI_Numba(cfg)
    mppi_controller.set_params(mppi_params)

    # Loop
    max_steps = 500
    xhist = np.zeros((max_steps+1, num_states))*np.nan
    uhist = np.zeros((max_steps, num_controls))*np.nan
    xhist[0] = x0

    vis_xlim = [-1, 8]
    vis_ylim = [-1, 6]

    plot_every_n = 15
    for t in range(max_steps):
        # Solve
        useq = mppi_controller.solve()
        u_curr = useq[0]
        phi, theta, psi = xhist[t, 6:9]
        gravity_vector_world = np.array([0, 0, -9.81*2.302499999999999])
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                       np.cos(phi)*np.cos(theta)]
        ])
        gravity_body = np.dot(R.T, gravity_vector_world)  # Rotate gravity to body frame
        u_curr[:3] += gravity_body
        uhist[t] = u_curr

        # Simulate state forward 
        xhist[t+1, :] = dynamics_update_sim(xhist[t, :], u_curr, cfg.dt)
        # print("x: ", xhist[t+1, :])
        print(t)
        # Update MPPI state (x0, useq)
        mppi_controller.shift_and_update(xhist[t+1], useq, num_shifts=1)

    # save actions for open loop
    np.save('actions_for_open_loop.npy', uhist)
    # Assuming xgoal is your goal position and it has appropriate values for each state
    x_goal, y_goal, z_goal = xgoal[:3]
    roll_goal, pitch_goal, yaw_goal = xgoal[6:9]
    
    fig, axs = plt.subplots(4, 3, figsize=(12, 9))  # Create 3 subplots, one for each series

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



    plt.tight_layout()  # Adjusts the subplots to fit in the figure area
    plt.show()
