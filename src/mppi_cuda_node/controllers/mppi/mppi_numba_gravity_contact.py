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


def dynamics_update_sim(x, u, dt):
  # The dynamics update for hexarotor
  # I_xx = 0.23038337
  # I_yy = 0.11771596
  # I_zz = 0.11392979
  I_xx = 0.115125971
  I_yy = 0.116524229
  I_zz = 0.230387752

  mass = 7.00
  g = 9.81

  
  x_next = x.copy()

  x_next[0] += dt * x[3] 
  x_next[1] += dt * x[4]
  x_next[2] += dt * x[5]
  
  x_next[3] += dt * ((1/mass) * u[0] - g * (np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) + np.sin(x[6]) * np.sin(x[8])) )
  x_next[4] += dt * ((1/mass) * u[1] - g * (np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) - np.sin(x[6]) * np.cos(x[8])) )
  x_next[5] += dt * ((1/mass) * u[2] - g * (np.cos(x[6]) * np.cos(x[7])) )

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
  return (1-np.float32(goal_reached))*dist2



@cuda.jit(device=True, fastmath=True)
def calculate_contact_force_moment_naiive(x, u, A, B, C, D, contact_normal_sq, contact_normal, k_p, k_d, k_f):
    """
    Computes the contact force and moment exerted on the aerial manipulator's end-effector
    using a spring-damper model.

    Args:
        x (array): State vector (position, velocity, orientation, angular velocity).
        u (array): Control input (forces and moments).
        A, B, C, D (float): Plane equation parameters.
        contact_normal_sq (float): Squared magnitude of the contact normal vector.
        contact_normal (array): Contact normal direction [nx, ny, nz].
        k_p (float): Stiffness coefficient.
        k_d (float): Normal damping coefficient.
        k_f (float): Friction damping coefficient.

    Returns:
        Contact forces (fx, fy, fz), contact velocities (vx, vy, vz), 
        and contact moments (mx, my, mz).
    """

    arm_length = 1.2  # Arm length from UAV body to end effector
    contact_threshold = 0.0  # Distance threshold for contact

    # Compute end-effector position
    ee_x = x[0] + arm_length * math.cos(x[7]) * math.cos(x[8])
    ee_y = x[1] + arm_length * math.cos(x[7]) * math.sin(x[8])
    ee_z = x[2] + arm_length * math.sin(x[7])

    # Distance from the contact plane
    dist_from_contact_plane = (A * ee_x + B * ee_y + C * ee_z + D) / math.sqrt(A**2 + B**2 + C**2)

    # Check if the end-effector is in contact
    contact_bitmask = 1.0 if dist_from_contact_plane < contact_threshold else 0.0

    # Compute penetration depth (distance into the surface)
    penetration = -dist_from_contact_plane * contact_bitmask  # Only apply force if penetration exists

    # Compute the velocity of the end effector
    v_ee_x = x[3] + arm_length * (-math.sin(x[7]) * math.cos(x[8]) * x[10] - math.cos(x[7]) * math.sin(x[8]) * x[11])
    v_ee_y = x[4] + arm_length * (-math.sin(x[7]) * math.sin(x[8]) * x[10] + math.cos(x[7]) * math.cos(x[8]) * x[11])
    v_ee_z = x[5] + arm_length * (math.cos(x[7]) * x[10])

    # Compute velocity component along the contact normal
    v_normal = (v_ee_x * contact_normal[0] + v_ee_y * contact_normal[1] + v_ee_z * contact_normal[2])

    # Compute tangential velocity (in X-Y plane relative to the surface)
    v_tangential_x = v_ee_x - v_normal * contact_normal[0]
    v_tangential_y = v_ee_y - v_normal * contact_normal[1]
    v_tangential_z = v_ee_z - v_normal * contact_normal[2]

    # Compute normal force using the spring-damper model
    f_contact_normal = (- k_p * penetration - k_d * v_normal) * contact_bitmask

    contact_force_x = f_contact_normal * contact_normal[0]
    contact_force_y = f_contact_normal * contact_normal[1]
    contact_force_z = f_contact_normal * contact_normal[2]

    # Compute friction force (damping in the tangential direction)
    contact_friction_x = - k_f * v_tangential_x * contact_bitmask
    contact_friction_y = - k_f * v_tangential_y * contact_bitmask
    contact_friction_z = - k_f * v_tangential_z * contact_bitmask

    # Combine forces: Normal + Friction
    contact_force_x += contact_friction_x
    contact_force_y += contact_friction_y
    contact_force_z += contact_friction_z

    # Compute contact moments due to contact forces
    contact_moment_x = (- (math.cos(x[7]) * math.sin(x[8]) * arm_length * contact_force_z + 
                           math.sin(x[7]) * arm_length * contact_force_y)) * contact_bitmask
    contact_moment_y = (- (-math.sin(x[7]) * arm_length * contact_force_x - 
                            math.cos(x[7]) * math.cos(x[8]) * arm_length * contact_force_z)) * contact_bitmask
    contact_moment_z = (- (math.cos(x[7]) * math.cos(x[8]) * arm_length * contact_force_y - 
                            math.cos(x[7]) * math.sin(x[8]) * arm_length * contact_force_x)) * contact_bitmask

    return (contact_force_x, contact_force_y, contact_force_z, 
            v_normal,  # Normal velocity
            contact_moment_x, contact_moment_y, contact_moment_z)

@cuda.jit(device=True, fastmath=True)
def dynamics_update(x, u, dt, contact_normal, inertia_mass, cf_out):
  # The dynamics update for hexarotor
  contact_normal = (-1, 0, 0)
  contact_normal_sq = 1
  A = -1
  B = 0
  C = 0
  D = 1.3
  ABC_sq = 1

  I_xx = inertia_mass[0]
  I_yy = inertia_mass[1]
  I_zz = inertia_mass[2]
  mass = inertia_mass[3]
  # working 1 50 16
  k_admittance = 1
  k_stiffness = 100
  c_damping = 16
  contact_force_x, contact_force_y, contact_force_z, \
  contact_velocity, \
  contact_moment_x, contact_moment_y, contact_moment_z = \
      calculate_contact_force_moment_naiive(x, u, A, B, C, D, contact_normal_sq, contact_normal, k_stiffness, c_damping, k_admittance)

  cf_out[0] = contact_force_x
  cf_out[1] = contact_force_y
  cf_out[2] = contact_force_z


  sin_phi = math.sin(x[6])
  cos_phi = math.cos(x[6])
  sin_theta = math.sin(x[7])
  cos_theta = math.cos(x[7])
  sin_psi = math.sin(x[8])
  cos_psi = math.cos(x[8])

  # Standard Z-Y-X Euler angles:
  R00 = cos_theta * cos_psi
  R01 = cos_theta * sin_psi
  R02 = -sin_theta

  R10 = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
  R11 = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
  R12 = sin_phi * cos_theta

  R20 = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
  R21 = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
  R22 = cos_phi * cos_theta
  
  # Transform the contact force from inertial to body frame.
  Fx_contact_body = R00 * contact_force_x + R10 * contact_force_y + R20 * contact_force_z
  Fy_contact_body = R01 * contact_force_x + R11 * contact_force_y + R21 * contact_force_z
  Fz_contact_body = R02 * contact_force_x + R12 * contact_force_y + R22 * contact_force_z

  g = 9.81

  fx_total = u[0] + Fx_contact_body 
  fy_total = u[1] + Fy_contact_body 
  fz_total = u[2] + Fz_contact_body 
  mx_total = u[3] #+ contact_moment_x 
  my_total = u[4] #+ contact_moment_y 
  mz_total = u[5] #+ contact_moment_z  

  
  x[0] += dt*x[3] 
  x[1] += dt*x[4]
  x[2] += dt*x[5]

  x[3] += dt*((1/mass) * fx_total - g * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi))
  x[4] += dt*((1/mass) * fy_total - g * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi))
  x[5] += dt*((1/mass) * fz_total - g * cos_phi * cos_theta)

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
      dynamics_update(x_curr, u_noisy, dt_d, CONTACT_NORMAL, inertia_mass_d, cf)

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
                    + cost_weights_d[12]*((u_nom[0]**2) + (u_nom[1]**2) + ((u_nom[2] - inertia_mass_d[3]*9.81)**2))\
                    + cost_weights_d[13]*((u_nom[3]**2) + (u_nom[4]**2) + (u_nom[5]**2)) + 10 * (cf[0] - 20) ** 2 
                    
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
  def get_state_rollout_across_control_noise(
          state_rollout_batch_d, # where to store results
          x0_d, 
          dt_d,
          noise_samples_d,
          vrange_d,
          wrange_d,
          u_prev_d,
          u_cur_d):
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
        dynamics_update(x_curr, u_nom, dt_d, CONTACT_NORMAL)

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
        dynamics_update(x_curr, u_noisy, dt_d, CONTACT_NORMAL)

        # Save state
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]

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
            num_control_rollouts=1024*4,
            num_controls=6,
            num_states=12,
            num_vis_state_rollouts=1,
            seed=1
        )
    x0 = np.array([-0.2,0, 0, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    # xgoal = np.array([2,-1, 3, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    # xgoal = np.array([2,-1, 3, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([0,0, 0.8, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    # xgoal = np.array([0.2,-0.2, 0.8, 0, 0, 0, 0.1, -0.1, -0.3, 0, 0, 0])
    xgoal = np.array([0.8,-0.2, 0.8, 0, 0, 0, 0.0, -0.0, -0.0, 0, 0, 0])
    
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
    max_steps = 300

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
        next_state = dynamics_update_rk4(current_state.copy(), mppi_u, np.zeros(3), mppi_params['dt'], mppi_params)
        # Zero-out the angular velocity components for the target
        # next_state[6:] = np.zeros(6)

        # next_state_filtered = self.lpf.filter(next_state.copy())
        # next_state_filtered[6:9] = np.clip(next_state_filtered[6:9], -0.1, 0.1)
        return next_state

    def dynamics_update_rk4_backup(state, control_inputs, dt, mppi_params):
        """
        A simple RK4 integration for the hexarotor dynamics.
        """
        # k1 = dynamics_update_sim(state, control_inputs, dt)
        # k2 = dynamics_update_sim(state + k1 / 2, control_inputs, dt) 
        # k3 = dynamics_update_sim(state + k2 / 2, control_inputs, dt)
        # k4 = dynamics_update_sim(state + k3, control_inputs, dt)
        
        k1 = hex_dynamics(state, control_inputs, mppi_params) * dt
        k2 = hex_dynamics(state + k1 / 2, control_inputs, mppi_params) * dt 
        k3 = hex_dynamics(state + k2 / 2, control_inputs, mppi_params) * dt
        k4 = hex_dynamics(state + k3, control_inputs, mppi_params) * dt
        next_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
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
        
        dynamics_update(local_x, u, dt, contact_normal, inertia_mass, cf_out)
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

      k1 = hex_dynamics(state, control_inputs, mppi_params) * dt
      k2 = hex_dynamics(state + k1 / 2, control_inputs, mppi_params) * dt 
      k3 = hex_dynamics(state + k2 / 2, control_inputs, mppi_params) * dt
      k4 = hex_dynamics(state + k3, control_inputs, mppi_params) * dt

      next_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
      # next_cf = cf + dt * (kf1 + 2 * kf2 + 2 * kf3 + kf4) / 6
      next_cf = cf + dt * (kf1)
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
          xhist[t+1, :], fhist[t+1, :] = dynamics_update_rk4(xhist[t, :], u_curr, fhist[t, :], cfg.dt, mppi_params)
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


    # # mpc target x
    # axs[4][0].plot(mpctargethist[:, 0], label='X')
    # axs[4][0].set_title('X')
    # axs[4][0].set_xlabel('Time Steps')
    # axs[4][0].set_ylabel('m')
    # axs[4][0].legend()
    # # mpc target y
    # axs[4][1].plot(mpctargethist[:, 1], label='Y')
    # axs[4][1].set_title('Y')
    # axs[4][1].set_xlabel('Time Steps')
    # axs[4][1].set_ylabel('m')
    # axs[4][1].legend()
    # # mpc target z
    # axs[4][2].plot(mpctargethist[:, 2], label='Z')
    # axs[4][2].set_title('Z')
    # axs[4][2].set_xlabel('Time Steps')
    # axs[4][2].set_ylabel('m')
    # axs[4][2].legend()

    # # mpc target r
    # axs[5][0].plot(mpctargethist[:, 6]*180/np.pi, label='roll')
    # axs[5][0].set_title('r')
    # axs[5][0].set_xlabel('Time Steps')
    # axs[5][0].set_ylabel('degrees')
    # axs[5][0].legend()
    # # mpc target p
    # axs[5][1].plot(mpctargethist[:, 7]*180/np.pi, label='pitch')
    # axs[5][1].set_title('p')
    # axs[5][1].set_xlabel('Time Steps')
    # axs[5][1].set_ylabel('degrees')
    # axs[5][1].legend()
    # # mpc target y
    # axs[5][2].plot(mpctargethist[:, 8]*180/np.pi, label='yaw')
    # axs[5][2].set_title('y')
    # axs[5][2].set_xlabel('Time Steps')
    # axs[5][2].set_ylabel('degrees')
    # axs[5][2].legend()



    plt.tight_layout()  # Adjusts the subplots to fit in the figure area
    plt.show()
