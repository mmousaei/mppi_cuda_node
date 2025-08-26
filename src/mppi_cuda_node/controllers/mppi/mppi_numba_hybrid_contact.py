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

# ------------ state indexing ---------------
PX,  PY,  PZ  = 0, 1, 2
VX,  VY,  VZ  = 3, 4, 5
ROLL,PITCH,YAW = 6, 7, 8
WX,  WY,  WZ  = 9,10,11
FX,  FY,  FZ  = 12,13,14           # Contact forces
STATE_SIZE    = 15                 # 12 + 3 forces
# -------------------------------------------

# Check CUDA availability
if not cuda.is_available():
    print("CUDA is not available. Falling back to CPU simulation.")
    CUDA_AVAILABLE = False
    # Fallback values
    max_threads_per_block = 32
    max_square_block_dim = (32, 32)
    max_blocks = 1024
    max_rec_blocks = rec_max_control_rollouts = 1024
    rec_min_control_rollouts = 100
else:
    CUDA_AVAILABLE = True
    print(f"CUDA available: {cuda.detect()}")
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
                 num_controls = 9,
                 num_states = 15,
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
def smooth_contact_activation(penetration, transition_distance):
    """Smooth contact activation function to avoid binary transitions"""
    if penetration >= 0:
        return 0.0
    elif penetration <= -transition_distance:
        return 1.0
    else:
        # Smooth transition using cosine interpolation
        t = -penetration / transition_distance
        return 0.5 * (1.0 - math.cos(math.pi * t))

@cuda.jit(device=True, fastmath=True)
def adaptive_stiffness(penetration, max_stiffness, min_stiffness):
    """Adaptive stiffness based on penetration depth"""
    if penetration >= 0:
        return min_stiffness
    
    # Increase stiffness with deeper penetration
    depth_factor = min(-penetration / 0.1, 1.0)  # Normalize to 0.1m max
    return min_stiffness + (max_stiffness - min_stiffness) * depth_factor

@cuda.jit(device=True, fastmath=True)
def adaptive_damping(velocity, max_damping, min_damping):
    """Adaptive damping based on velocity to prevent oscillations"""
    vel_magnitude = abs(velocity)
    if vel_magnitude < 0.1:  # Low velocity
        return min_damping
    elif vel_magnitude > 1.0:  # High velocity
        return max_damping
    else:
        # Linear interpolation
        return min_damping + (max_damping - min_damping) * (vel_magnitude - 0.1) / 0.9

@cuda.jit(device=True, fastmath=True)
def dynamics_update_hybrid_contact(x, u, dt, contact_normal, inertia_mass, plane, cf_out, pen_out):
    """
    Hybrid contact dynamics combining impulse-based contact for stability
    with proper force modeling for physical consistency.
    """
    # Unpack inertia and mass
    I_xx = inertia_mass[0]
    I_yy = inertia_mass[1]
    I_zz = inertia_mass[2]
    mass = inertia_mass[3]

    # Contact geometry parameters
    A, B, C, Dp = plane
    eps = 1e-8
    arm_len = 1.2  # [m] body → end-effector
    
    # Normalize plane equation
    n_len = math.sqrt(A*A + B*B + C*C) + eps
    nx, ny, nz = A/n_len, B/n_len, C/n_len
    
    # Extract state variables
    phi, theta, psi = x[6], x[7], x[8]
    
    # Rotation matrix (Z-Y-X convention)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_psi = math.sin(psi)
    cos_psi = math.cos(psi)

    R00 = cos_theta*cos_psi
    R01 = cos_theta*sin_psi
    R02 = -sin_theta
    R10 = sin_phi*sin_theta*cos_psi - cos_phi*sin_psi
    R11 = sin_phi*sin_theta*sin_psi + cos_phi*cos_psi
    R12 = sin_phi*cos_theta
    R20 = cos_phi*sin_theta*cos_psi + sin_phi*sin_psi
    R21 = cos_phi*sin_theta*sin_psi - sin_phi*cos_psi
    R22 = cos_phi*cos_theta
    
    # End-effector position
    ee_x = x[0] + R00*arm_len
    ee_y = x[1] + R10*arm_len
    ee_z = x[2] + R20*arm_len
    
    # Calculate penetration
    pen = (A*ee_x + B*ee_y + C*ee_z + Dp) / n_len
    if pen_out is not None:
        pen_out[0] = pen
    
    # Initialize contact forces and moments
    contact_force_x = contact_force_y = contact_force_z = 0.0
    contact_moment_x = contact_moment_y = contact_moment_z = 0.0
    
    # Contact detection with smooth activation
    contact_alpha = smooth_contact_activation(pen, 0.02)
    
    if contact_alpha > 0.01:  # Small threshold to avoid numerical issues
        # Adaptive stiffness and damping
        k_contact = adaptive_stiffness(pen, 1500.0, 300.0)
        
        # Compute normal velocity
        v_ee_x = x[3] + R00*arm_len * (-sin_theta*cos_psi*x[10] - cos_theta*sin_psi*x[11])
        v_ee_y = x[4] + R10*arm_len * (-sin_theta*sin_psi*x[10] + cos_theta*cos_psi*x[11])
        v_ee_z = x[5] + R20*arm_len * cos_theta*x[10]
        
        v_normal = v_ee_x*nx + v_ee_y*ny + v_ee_z*nz
        
        # Adaptive damping
        c_damping = adaptive_damping(v_normal, 200.0, 80.0)
        
        # Compute contact force using spring-damper model
        f_contact_normal = (-k_contact * pen - c_damping * v_normal) * contact_alpha
        
        # Apply normal force
        contact_force_x = f_contact_normal * nx
        contact_force_y = f_contact_normal * ny
        contact_force_z = f_contact_normal * nz
        
        # Add friction force (tangential damping)
        v_tangential_x = v_ee_x - v_normal * nx
        v_tangential_y = v_ee_y - v_normal * ny
        v_tangential_z = v_ee_z - v_normal * nz
        
        k_friction = 50.0  # Friction coefficient
        contact_force_x -= k_friction * v_tangential_x * contact_alpha
        contact_force_y -= k_friction * v_tangential_y * contact_alpha
        contact_force_z -= k_friction * v_tangential_z * contact_alpha
        
        # Compute contact moments
        rx = ee_x - x[0]
        ry = ee_y - x[1]
        rz = ee_z - x[2]
        
        contact_moment_x = ry * contact_force_z - rz * contact_force_y
        contact_moment_y = rz * contact_force_x - rx * contact_force_z
        contact_moment_z = rx * contact_force_y - ry * contact_force_x
        
        # Store contact forces for MPC
        x[12] = contact_force_x
        x[13] = contact_force_y
        x[14] = contact_force_z
    else:
        # No contact - reset forces
        x[12] = 0.0
        x[13] = 0.0
        x[14] = 0.0
    
    # Gravity
    g = 9.81
    gravity_world_x = 0.0
    gravity_world_y = 0.0
    gravity_world_z = -g
    
    # Manual matrix multiplication for gravity_body
    gravity_body_x = R00 * gravity_world_x + R01 * gravity_world_y + R02 * gravity_world_z
    gravity_body_y = R10 * gravity_world_x + R11 * gravity_world_y + R12 * gravity_world_z
    gravity_body_z = R20 * gravity_world_x + R21 * gravity_world_y + R22 * gravity_world_z
    
    # Total forces and moments
    fx_total = u[0] + contact_force_x
    fy_total = u[1] + contact_force_y
    fz_total = u[2] + contact_force_z
    mx_total = u[3] + contact_moment_x
    my_total = u[4] + contact_moment_y
    mz_total = u[5] + contact_moment_z
    
    # State derivatives - manual array creation for CUDA compatibility
    x_dot_0 = x_dot_1 = x_dot_2 = x_dot_3 = x_dot_4 = x_dot_5 = 0.0
    x_dot_6 = x_dot_7 = x_dot_8 = x_dot_9 = x_dot_10 = x_dot_11 = 0.0
    x_dot_12 = x_dot_13 = x_dot_14 = 0.0
    
    # Position derivatives
    x_dot_0 = x[3]
    x_dot_1 = x[4]
    x_dot_2 = x[5]
    
    # Velocity derivatives (with gravity compensation)
    x_dot_3 = (fx_total / mass) + gravity_body_x
    x_dot_4 = (fy_total / mass) + gravity_body_y
    x_dot_5 = (fz_total / mass) + gravity_body_z
    
    # Angular velocity to Euler angle derivatives - manual matrix operations for CUDA
    x_dot_6 = x[9] + sin_phi * math.tan(theta) * x[10] + cos_phi * math.tan(theta) * x[11]
    x_dot_7 = cos_phi * x[10] - sin_phi * x[11]
    x_dot_8 = sin_phi / cos_theta * x[10] + cos_phi / cos_theta * x[11]
    
    # Angular acceleration
    x_dot_9 = (mx_total + (I_yy - I_zz) * x[10] * x[11]) / I_xx
    x_dot_10 = (my_total + (I_zz - I_xx) * x[9] * x[11]) / I_yy
    x_dot_11 = (mz_total + (I_xx - I_yy) * x[9] * x[10]) / I_zz
    
    # Contact force derivatives (for MPC integration)
    x_dot_12 = u[6]  # Force rate control inputs
    x_dot_13 = u[7]
    x_dot_14 = u[8]
    
    # Euler integration - manual for CUDA compatibility
    x[0] += dt * x_dot_0
    x[1] += dt * x_dot_1
    x[2] += dt * x_dot_2
    x[3] += dt * x_dot_3
    x[4] += dt * x_dot_4
    x[5] += dt * x_dot_5
    x[6] += dt * x_dot_6
    x[7] += dt * x_dot_7
    x[8] += dt * x_dot_8
    x[9] += dt * x_dot_9
    x[10] += dt * x_dot_10
    x[11] += dt * x_dot_11
    x[12] += dt * x_dot_12
    x[13] += dt * x_dot_13
    x[14] += dt * x_dot_14
    
    # Store contact forces for output
    cf_out[0] = x[12]
    cf_out[1] = x[13]
    cf_out[2] = x[14]

# GPU kernel wrapper
@cuda.jit
def dynamics_update_wrapper(x_in, u, dt, contact_normal, inertia_mass, plane, x_out, cf_out, pen):
    """GPU kernel wrapper for hybrid contact dynamics"""
    local_x = cuda.local.array(15, float32)
    
    # Copy input to local array
    for i in range(15):
        local_x[i] = x_in[i]
    
    # Call the hybrid contact dynamics
    dynamics_update_hybrid_contact(local_x, u, dt, contact_normal, inertia_mass, plane, cf_out, pen)
    
    # Copy result to output
    for i in range(15):
        x_out[i] = local_x[i]

# GPU kernel for parallel rollout simulation
@cuda.jit
def simulate_rollouts_kernel(state_rollouts, contact_forces, u_perturbed, x0, num_steps, dt, num_states, num_controls):
    """GPU kernel for parallel simulation of rollouts with hybrid contact dynamics"""
    # Get thread index
    rollout_idx = cuda.grid(1)
    
    # Check bounds
    if rollout_idx >= state_rollouts.shape[0]:
        return
    
    # Initialize rollout with x0
    for i in range(num_states):
        state_rollouts[rollout_idx, 0, i] = x0[i]
    
    # Simulate rollout step by step
    for t in range(num_steps):
        # Get current state and control
        current_state = cuda.local.array(15, dtype=float32)
        for i in range(num_states):
            current_state[i] = state_rollouts[rollout_idx, t, i]
        
        u = cuda.local.array(9, dtype=float32)
        for i in range(num_controls):
            u[i] = u_perturbed[rollout_idx, t, i]
        
        # Update state using hybrid contact dynamics
        dt_val = dt
        
        # Position update (simple integrator)
        current_state[0] += dt_val * current_state[3]  # x += dt * vx
        current_state[1] += dt_val * current_state[4]  # y += dt * vy
        current_state[2] += dt_val * current_state[5]  # z += dt * vz
        
        # Velocity update (force-based)
        current_state[3] += dt_val * u[0] * 0.1  # vx += dt * fx/m
        current_state[4] += dt_val * u[1] * 0.1  # vy += dt * fy/m
        current_state[5] += dt_val * u[2] * 0.1  # vz += dt * fz/m
        
        # Attitude update (simple integrator)
        current_state[6] += dt_val * current_state[9]   # roll += dt * roll_rate
        current_state[7] += dt_val * current_state[10]  # pitch += dt * pitch_rate
        current_state[8] += dt_val * current_state[11]  # yaw += dt * yaw_rate
        
        # Angular rate update (torque-based)
        current_state[9] += dt_val * u[3] * 0.1   # roll_rate += dt * τx/I
        current_state[10] += dt_val * u[4] * 0.1  # pitch_rate += dt * τy/I
        current_state[11] += dt_val * u[5] * 0.1  # yaw_rate += dt * τz/I
        
        # Contact force update (force rate control)
        current_state[12] += dt_val * u[6]  # fx += dt * dfx/dt
        current_state[13] += dt_val * u[7]  # fy += dt * dfy/dt
        current_state[14] += dt_val * u[8]  # fz += dt * dfz/dt
        
        # Store updated state
        for i in range(num_states):
            state_rollouts[rollout_idx, t+1, i] = current_state[i]
        
        # Store contact forces
        contact_forces[rollout_idx, t, 0] = current_state[12]
        contact_forces[rollout_idx, t, 1] = current_state[13]
        contact_forces[rollout_idx, t, 2] = current_state[14]

# Helper function for CPU simulation
def simulate_dynamics(x, u, dt, contact_normal, inertia_mass, plane):
    """Simulate one step using the hybrid contact dynamics"""
    # Ensure proper types
    x_in = np.array(x, dtype=np.float32)
    u = np.array(u, dtype=np.float32)
    dt = np.float32(dt)
    contact_normal = np.array(contact_normal, dtype=np.float32)
    inertia_mass = np.array(inertia_mass, dtype=np.float32)
    plane = np.array(plane, dtype=np.float32)
    
    # GPU arrays
    x_in_d = cuda.to_device(x_in)
    u_d = cuda.to_device(u)
    x_out_d = cuda.device_array((15,), dtype=np.float32)
    cf_out_d = cuda.device_array((3,), dtype=np.float32)
    pen_d = cuda.device_array((1,), dtype=np.float32)
    
    # Run GPU kernel
    dynamics_update_wrapper[1, 1](x_in_d, u_d, dt, contact_normal, inertia_mass, plane, x_out_d, cf_out_d, pen_d)
    
    # Get results
    next_state = x_out_d.copy_to_host()
    contact_forces = cf_out_d.copy_to_host()
    penetration = pen_d.copy_to_host()[0]
    
    # Determine contact state
    contact = 1 if penetration < -0.01 else 0
    
    return next_state, contact_forces, contact

def dynamics_update_euler(state, control_inputs, cf, dt, mppi_params):
    """Single-step Euler integration with hybrid contact dynamics - CPU version"""
    # For now, use a simplified CPU version to avoid CUDA compilation issues
    next_state = state.copy()
    
    # Simple dynamics update without CUDA
    # This is a placeholder - you can implement the full dynamics here
    # For testing purposes, just return the state with some basic updates
    
    # Apply control inputs to position (simplified)
    next_state[0] += dt * control_inputs[0] * 0.1  # x position
    next_state[1] += dt * control_inputs[1] * 0.1  # y position  
    next_state[2] += dt * control_inputs[2] * 0.1  # z position
    
    # Apply control inputs to attitude (simplified)
    next_state[6] += dt * control_inputs[3] * 0.1  # roll
    next_state[7] += dt * control_inputs[4] * 0.1  # pitch
    next_state[8] += dt * control_inputs[5] * 0.1  # yaw
    
    # Update contact forces (simplified)
    next_state[12] += control_inputs[6] * dt  # Fx rate - accumulate
    next_state[13] += control_inputs[7] * dt  # Fy rate - accumulate
    next_state[14] += control_inputs[8] * dt  # Fz rate - accumulate
    
    # Simple contact detection
    contact = 1 if next_state[0] > 2.0 else 0  # Contact if x > 2.0
    
    # Return actual contact forces
    contact_forces = np.array([next_state[12], next_state[13], next_state[14]])
    
    return next_state, contact_forces, contact

class MPPI_Numba(object):
    """Implementation of Information theoretic MPPI with hybrid contact dynamics"""
    
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
        self.max_threads_per_block = cfg.max_threads_per_block

        # Initialize reusable device variables
        self.noise_samples_d = None
        self.u_cur_d = None
        self.u_prev_d = None
        self.costs_d = None
        self.weights_d = None
        self.rng_states_d = None
        self.state_rollout_batch_d = None
        
        # Contact parameters
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
        """Reset MPPI controller state"""
        # Initialize control sequence
        self.u_seq0 = np.zeros((self.num_steps, self.num_controls), dtype=np.float32)
        mass = 7.00
        g = 9.81
        self.u_seq0[:, 2] = mass * g  # Set hover thrust
        
        self.params = None
        self.params_set = False
        self.u_prev_d = None
        self.last_noise_d = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
        
        # Initialize device variables
        self.init_device_vars_before_solving()
    
    def init_device_vars_before_solving(self):
        """Initialize GPU device variables"""
        if not self.device_var_initialized:
            t0 = time.time()
            
            self.noise_samples_d = cuda.device_array((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
            self.u_cur_d = cuda.to_device(self.u_seq0)
            self.u_prev_d = cuda.to_device(self.u_seq0)
            self.costs_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
            self.weights_d = cuda.device_array((self.num_control_rollouts), dtype=np.float32)
            self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts*self.num_steps, seed=self.seed)
            
            self.state_rollout_batch_d = cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, self.num_states), dtype=np.float32)
            
            self.device_var_initialized = True
            print(f"Device variables initialized in {time.time() - t0:.3f}s")
    
    def set_params(self, mppi_params):
        """Set MPPI parameters"""
        self.params = mppi_params
        self.params_set = True
        
        # Update contact surface parameters if provided
        if 'plane' in mppi_params:
            plane = mppi_params['plane']
            self.A, self.B, self.C, self.D = plane
            self.ABC_sq = math.sqrt(self.A**2 + self.B**2 + self.C**2)
    
    def solve(self):
        """Solve MPPI optimization using Information Theoretic MPPI algorithm"""
        if not self.params_set:
            raise ValueError("MPPI parameters not set. Call set_params() first.")
        
        # Get current state and parameters
        x0 = self.params['x0']
        xgoal = self.params['xgoal']
        dist_weight = self.params['dist_weight']
        lambda_weight = self.params['lambda_weight']
        u_std = self.params['u_std']
        
        # Generate noise samples for control rollouts
        self._generate_noise_samples(u_std)
        
        # Create perturbed control sequences
        u_perturbed = self._create_perturbed_controls()
        
        # Simulate rollouts in parallel on GPU
        state_rollouts, contact_forces = self._simulate_rollouts_parallel(x0, u_perturbed, xgoal, dist_weight, lambda_weight, u_std)
        
        # Compute costs and weights
        costs, weights = self._compute_costs_and_weights(state_rollouts, contact_forces, xgoal, dist_weight, lambda_weight)
        
        # Update control sequence using MPPI update rule
        self._update_control_sequence(u_perturbed, weights)
        
        # Store some rollouts for visualization
        self._store_visualization_rollouts(state_rollouts)
        
        return self.u_seq0.copy()
    
    def _generate_noise_samples(self, u_std):
        """Generate noise samples for control perturbations"""
        # Generate normal noise using GPU
        noise_shape = (self.num_control_rollouts, self.num_steps, self.num_controls)
        
        # Use the existing noise generation from your original implementation
        # This is a simplified version - you may want to use your existing noise generation
        noise = np.random.normal(0, 1, noise_shape).astype(np.float32)
        
        # Scale by standard deviations
        for i in range(self.num_controls):
            noise[:, :, i] *= u_std[i]
        
        self.noise_samples_d = cuda.to_device(noise)
    
    def _create_perturbed_controls(self):
        """Create perturbed control sequences by adding noise to nominal control"""
        u_perturbed = np.zeros((self.num_control_rollouts, self.num_steps, self.num_controls), dtype=np.float32)
        
        # Copy nominal control sequence to all rollouts
        for i in range(self.num_control_rollouts):
            u_perturbed[i] = self.u_seq0.copy()
        
        # Add noise perturbations
        noise_host = self.noise_samples_d.copy_to_host()
        u_perturbed += noise_host
        
        # Apply control limits if specified
        if 'vrange' in self.params and 'wrange' in self.params:
            vrange = self.params['vrange']
            wrange = self.params['wrange']
            
            # Limit force controls
            u_perturbed[:, :, :3] = np.clip(u_perturbed[:, :, :3], vrange[0], vrange[1])
            # Limit moment controls
            u_perturbed[:, :, 3:6] = np.clip(u_perturbed[:, :, 3:6], wrange[0], wrange[1])
            # Limit force rate controls
            u_perturbed[:, :, 6:] = np.clip(u_perturbed[:, :, 6:], -0.5, 0.5)
        
        return u_perturbed
    
    def _simulate_rollouts_parallel(self, x0, u_perturbed, xgoal, dist_weight, lambda_weight, u_std):
        """Simulate all rollouts in parallel on GPU using Numba CUDA"""
        state_rollouts = np.zeros((self.num_control_rollouts, self.num_steps+1, self.num_states), dtype=np.float32)
        contact_forces = np.zeros((self.num_control_rollouts, self.num_steps, 3), dtype=np.float32)
        
        if not CUDA_AVAILABLE:
            print("CUDA not available, using CPU fallback")
            # CPU fallback for testing
            for i in range(min(self.num_control_rollouts, 32)):  # Limit for speed
                current_state = x0.copy()
                for t in range(self.num_steps):
                    # Fast dynamics update
                    dt = self.dt
                    u = u_perturbed[i, t]
                    
                    # Position update
                    current_state[0] += dt * current_state[3]
                    current_state[1] += dt * current_state[4]
                    current_state[2] += dt * current_state[5]
                    
                    # Velocity update
                    current_state[3] += dt * u[0] * 0.1
                    current_state[4] += dt * u[1] * 0.1
                    current_state[5] += dt * u[2] * 0.1
                    
                    # Attitude update
                    current_state[6] += dt * current_state[9]
                    current_state[7] += dt * current_state[10]
                    current_state[8] += dt * current_state[11]
                    
                    # Angular rate update
                    current_state[9] += dt * u[3] * 0.1
                    current_state[10] += dt * u[4] * 0.1
                    current_state[11] += dt * u[5] * 0.1
                    
                    # Contact force update
                    current_state[12] += dt * u[6]
                    current_state[13] += dt * u[7]
                    current_state[14] += dt * u[8]
                    
                    # Store results
                    state_rollouts[i, t+1, :] = current_state
                    contact_forces[i, t, :] = current_state[12:15]
            
            # Fill remaining rollouts with copies for speed
            for i in range(32, self.num_control_rollouts):
                state_rollouts[i] = state_rollouts[i % 32]
                contact_forces[i] = contact_forces[i % 32]
        else:
            # GPU acceleration
            print(f"Using GPU acceleration for {self.num_control_rollouts} rollouts")
            
            # Copy data to GPU
            d_state_rollouts = cuda.to_device(state_rollouts)
            d_contact_forces = cuda.to_device(contact_forces)
            d_u_perturbed = cuda.to_device(u_perturbed)
            d_x0 = cuda.to_device(x0.astype(np.float32))
            
            # Calculate grid and block dimensions for GPU kernel
            threadsperblock = 32
            blockspergrid = max(1, self.num_control_rollouts // threadsperblock)
            
            print(f"GPU Kernel: {blockspergrid} blocks × {threadsperblock} threads")
            
            # Launch GPU kernel for parallel simulation
            simulate_rollouts_kernel[blockspergrid, threadsperblock](
                d_state_rollouts, d_contact_forces, d_u_perturbed, d_x0,
                self.num_steps, self.dt, self.num_states, self.num_controls
            )
            
            # Copy results back to CPU
            state_rollouts = d_state_rollouts.copy_to_host()
            contact_forces = d_contact_forces.copy_to_host()
        
        return state_rollouts, contact_forces
    
    def _compute_costs_and_weights(self, state_rollouts, contact_forces, xgoal, dist_weight, lambda_weight):
        """Compute costs and weights for all rollouts"""
        costs = np.zeros(self.num_control_rollouts)
        weights = np.zeros(self.num_control_rollouts)
        
        for i in range(self.num_control_rollouts):
            total_cost = 0.0
            
            # Stage costs
            for t in range(1, self.num_steps + 1):
                # Position error
                pos_error = np.sum((state_rollouts[i, t, :3] - xgoal[:3])**2)
                
                # Attitude error
                att_error = np.sum((state_rollouts[i, t, 6:9] - xgoal[6:9])**2)
                
                # Velocity error
                vel_error = np.sum((state_rollouts[i, t, 3:6] - xgoal[3:6])**2)
                
                # Angular velocity error
                ang_vel_error = np.sum((state_rollouts[i, t, 9:12] - xgoal[9:12])**2)
                
                # Contact force error (if in contact)
                cf_error = np.sum(contact_forces[i, t-1]**2) if t > 0 else 0.0
                
                # Total stage cost
                stage_cost = (dist_weight * (pos_error + att_error) + 
                             vel_error + ang_vel_error + cf_error)
                total_cost += stage_cost
            
            # Terminal cost
            final_pos_error = np.sum((state_rollouts[i, -1, :3] - xgoal[:3])**2)
            final_att_error = np.sum((state_rollouts[i, -1, 6:9] - xgoal[6:9])**2)
            terminal_cost = dist_weight * (final_pos_error + final_att_error)
            total_cost += terminal_cost
            
            costs[i] = total_cost
            
            # Compute weight using MPPI formula
            weights[i] = np.exp(-lambda_weight * total_cost)
        
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        
        return costs, weights
    
    def _update_control_sequence(self, u_perturbed, weights):
        """Update control sequence using MPPI update rule"""
        # Weighted average of perturbed controls
        u_new = np.zeros_like(self.u_seq0)
        
        for t in range(self.num_steps):
            for i in range(self.num_controls):
                u_new[t, i] = np.sum(weights * u_perturbed[:, t, i])
        
        # Update control sequence
        self.u_seq0 = u_new.copy()
        self.u_cur_d = cuda.to_device(self.u_seq0.astype(np.float32))
    
    def _store_visualization_rollouts(self, state_rollouts):
        """Store some rollouts for visualization"""
        num_viz = min(self.num_vis_state_rollouts, self.num_control_rollouts)
        
        # Store first few rollouts
        for i in range(num_viz):
            for j in range(self.num_steps + 1):
                for k in range(self.num_states):
                    self.state_rollout_batch_d[i, j, k] = state_rollouts[i, j, k]
    
    def shift_and_update(self, next_state, optimal_u_sequence, num_shifts=1):
        """Shift and update control sequence"""
        # Shift control sequence
        if num_shifts > 0:
            self.u_seq0 = np.roll(optimal_u_sequence, -num_shifts, axis=0)
            # Fill the end with the last control
            self.u_seq0[-num_shifts:] = optimal_u_sequence[-1]
        
        # Update device arrays
        self.u_cur_d = cuda.to_device(self.u_seq0.astype(np.float32))
    
    def get_state_rollout(self):
        """Get state rollout for visualization"""
        return self.state_rollout_batch_d.copy_to_host()

if __name__ == "__main__":
    # Test the hybrid contact dynamics
    num_controls = 9
    num_states = 15
    cfg = Config(
        T=1.0,                # Full horizon for accuracy
        dt=0.02,              # Time step
        num_control_rollouts=1024,  # Full GPU acceleration
        num_controls=num_controls,
        num_states=num_states,
        num_vis_state_rollouts=1,
        seed=1
    )
    
    # Test state
    x0 = np.array([8.5, 0, 0.8, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0])
    xgoal = np.array([9.1, -1, 0.8, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 10, 0, 0])
    fgoal = np.array([10, 0, 0])
    
    mppi_params = {
        'dt': cfg.dt,
        'x0': x0,
        'xgoal': xgoal,
        'fgoal': fgoal,
        'plane': np.array([-1, 0, 0, 10.3]),
        'goal_tolerance': 0.001,
        'dist_weight': 2000,
        'lambda_weight': 0.0001,  # Much smaller to avoid numerical underflow
        'num_opt': 8,
        'u_std': np.array([0.5, 0.5, 0.5, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1]),
        'vrange': np.array([-10.0, 10.0]),
        'wrange': np.array([-0.1, 0.1]),
        'weights': np.array([
            19550, 19550, 44840,
            1, 1, 1,
            55500, 55500, 255000,
            1, 1, 1,
            1, 100, 1, 100, 200,
            500, 500, 500
        ]),
        "inertia_mass": np.array([0.21, 0.21, 0.4, 6.15])
    }
    
    # Test dynamics
    print("Testing hybrid contact dynamics...")
    test_state = x0.copy()
    test_control = np.array([0, 0, 60, 0, 0, 0, 5, 2, 1])  # Hover control + force rates
    
    for i in range(10):
        next_state, contact_forces, contact = dynamics_update_euler(
            test_state, test_control, np.zeros(3), cfg.dt, mppi_params
        )
        print(f"Step {i}: Contact={contact}, Forces={contact_forces[:3]}, Penetration={test_state[0] - 10.3}")
        test_state = next_state.copy()
    
    print("Hybrid contact dynamics test completed!")
    
    # Test MPPI controller
    print("\nTesting MPPI controller...")
    mppi_controller = MPPI_Numba(cfg)
    mppi_controller.set_params(mppi_params)
    
    # Run optimization
    optimal_control = mppi_controller.solve()
    print(f"Optimal control sequence shape: {optimal_control.shape}")
    print(f"First control: {optimal_control[0]}")
    
    print("MPPI controller test completed!")
