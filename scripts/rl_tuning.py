import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from mppi_numba_gravity import MPPI_Numba, Config
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Box
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
class MPPITuningEnv(Env):
    def __init__(self):
        # Define action and observation spaces
        self.action_space = Box(low=-1, high=1, shape=(24,), dtype=np.float32)  # 17 weights, 6 u_std, 1 lambda_weight
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)  # State of the hexarotor

        # Initialize MPPI parameters
        self.num_states = 12
        self.num_controls = 6
        self.cfg = Config(T = 0.6,
            dt = 0.02,
            num_control_rollouts = 1024,#int(2e4), # Same as number of blocks, can be more than 1024
            num_controls = self.num_controls,
            num_states = self.num_states,
            num_vis_state_rollouts = 1,
            seed = 1)
        
        self.dt = 0.02
        self.max_steps = 100
        self.step_count = 0
        self.x = np.array([2, -1, 3, 0, 0, 0, 0.1, -0.1, 0.3, 0, 0, 0])  # Initial state
        self.xgoal = np.array([2, -1, 3, 0, 0, 0, 0.1, -0.1, 0.3, 0, 0, 0])  # Goal state
        self.mppi_params = dict(
            # Task specification
            dt=self.cfg.dt,
            x0=self.x, # Start state
            xgoal=self.xgoal, # Goal position

            # For risk-aware min time planning
            goal_tolerance=0.001,
            dist_weight=2000, #  Weight for dist-to-goal cost.
            # dist_weights = np.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0]),
            lambda_weight=10, # Temperature param in MPPI
            num_opt=2, # Number of steps in each solve() function call.

            # Control and sample specification
            u_std=np.array([1, 1, 1, 0.01, 0.01, 0.006]), # Noise std for sampling linear and angular velocities.
            vrange = np.array([-60.0, 60.0]), # Linear velocity range.
            wrange=np.array([-0.1, 0.1]), # Angular velocity range.
            # weights = np.array([150, 150, 300, 15, 1500, 1500, 3000, 100, 1, 5, 5, 1, 100]), # w_pose_x, w_pose_y, w_pose_z, w_vel, w_att_roll, w_att_pitch, w_att_yaw, w_omega, w_cont, w_cont_m, w_cont_f, w_cont_M, w_terminal
            weights = np.array([4550, 4*4550, 5300, 150, 4*150, 150, 5*75000, 5*35000, 2*85000, 500, 500, 1000, 1, 5, 5, 1, 500]), # w_pose_x, w_pose_y, w_pose_z, w_vel_x, w_vel_y, w_vel_z, w_att_roll, w_att_pitch, w_att_yaw, w_omega, w_cont, w_cont_m, w_cont_f, w_cont_M, w_terminal
            inertia_mass = np.array([0.115125971, 0.116524229, 0.230387752, 7.00]) # I_xx, I_yy, I_zz, mass
        )
        self.mppi_controller = MPPI_Numba(self.cfg)
        self.mppi_controller.set_params(self.mppi_params)
        # Initialize weights
        self.weights = np.array([4550, 4 * 4550, 5300, 150, 4 * 150, 150, 5 * 75000, 
                                5 * 35000, 2 * 85000, 500, 500, 1000, 1, 5, 5, 1, 500], dtype=np.float64)
        self.u_std = np.array([1, 1, 1, 0.01, 0.01, 0.006], dtype=np.float64)  # Noise standard deviations
        self.lambda_weight = 10.0  # Initial lambda weight

        # Other initializations...
        self.best_reward = -np.inf  # Initialize best reward to negative infinity
        self.best_weights = None
        self.best_u_std = None
        self.best_lambda_weight = None
    
    def step(self, action):
        """
        Perform one RL environment step, where the agent sets MPPI parameters,
        and a 300-step simulation is run to evaluate performance.
        """
        # Scale action to adjust MPPI parameters
        weight_adjust = action[:17] * 10  # Scale all 17 weights
        u_std_adjust = np.zeros(6)  # Initialize an array for u_std adjustments
        u_std_adjust[:3] = action[17:20] * 0.1  # Scale first 3 elements by 0.1
        u_std_adjust[3:6] = action[20:23] * 0.001  # Scale second 3 elements by 0.001   
        lambda_adjust = action[-1] * 1  # Last action adjusts lambda weight
        
        # Apply adjustments
        self.weights = np.clip(self.weights + weight_adjust, 1e-3, 1e6)  # Avoid zero or negative weights
        # Split u_std into two parts and clip separately
        self.u_std[:3] = np.clip(self.u_std[:3] + u_std_adjust[:3], 0.01, 10)  # Clip first 3 elements between 0.01 and 10
        self.u_std[3:6] = np.clip(self.u_std[3:6] + u_std_adjust[3:6], 0.00001, 0.1)  # Clip second 3 elements between 0.00001 and 0.1
        self.lambda_weight = np.clip(self.lambda_weight + lambda_adjust, 0.1, 100)  # Ensure positive lambda

        print("weights: ", self.weights)
        print("u_std: ", self.u_std)
        print("lambda_weight: ", self.lambda_weight)
        self.mppi_params.update({
            "weights": self.weights,
            "u_std": self.u_std,
            "lambda_weight": self.lambda_weight,
        })
        self.mppi_controller.set_params(self.mppi_params)

        
        # Simulate for 300 steps
        num_simulation_steps = 300
        cumulative_position_error = 0
        cumulative_orientation_error = 0
        control_variance = 0
        prev_u_curr = None  # To calculate control jitter

        for _ in range(num_simulation_steps):
            # Compute MPPI control
            useq = self.mppi_controller.solve()
            u_curr = useq[0]
            # Update dynamics
            self.x = dynamics_update_sim(self.x, u_curr, self.dt)
            if np.any(np.isnan(self.x)):
                print("NaN detected in state, resetting environment.")
                self.reset()  # Reset the environment
                reward = -1e9  # Large penalty for NaN
                return self.x, reward, True, {"info": "NaN detected and environment reset"}

            # Position error
            position_error = np.linalg.norm(self.x[:3] - self.xgoal[:3])
            cumulative_position_error += position_error

            # Orientation error (roll, pitch, yaw)
            orientation_error = np.linalg.norm(self.x[6:9] - self.xgoal[6:9])
            cumulative_orientation_error += orientation_error

            # Jitter (variance of control inputs)
            if prev_u_curr is not None:
                control_variance += np.sum((u_curr - prev_u_curr) ** 2)
            prev_u_curr = u_curr
            self.mppi_controller.shift_and_update(self.x, useq, num_shifts=1)

        # Compute final reward (negative penalty for errors and jitter)
        reward = (
            - cumulative_position_error  # Penalize trajectory tracking error
            - cumulative_orientation_error * 0.5  # Penalize orientation error
            - control_variance * 0.01  # Penalize jitter
        )

        # Done if tracking error is small enough
        done = cumulative_position_error < 10 and cumulative_orientation_error < 5
        self.step_count += 1  # Increment step count
        print("cumulative_position_error", cumulative_position_error)
        print("cumulative_orientation_error", cumulative_orientation_error)
        # Check if step count exceeds maximum steps
        if self.step_count >= self.max_steps:
            print("Maximum steps reached. Ending episode.")
            done = True
            reward += -1e5  # Penalty for exceeding maximum steps
            return self.x, reward, done, {"info": "Max steps reached"}
        info = {
            "cumulative_position_error": cumulative_position_error,
            "cumulative_orientation_error": cumulative_orientation_error,
            "control_variance": control_variance,
        }
        # Check if this step's reward is the best so far
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_weights = self.weights.copy()
            self.best_u_std = self.u_std.copy()
            self.best_lambda_weight = self.lambda_weight

        return self.x, reward, done, info


    
    def reset(self):
        # Reset environment state and parameters
        self.x = np.zeros(self.num_states)
        self.step_count = 0
        return self.x



# Train RL Agent
if __name__ == "__main__":
    env = DummyVecEnv([lambda: MPPITuningEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    # Test the tuned MPPI controller
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    # After training
    env_instance = env.envs[0]  # Access the first environment
    best_weights = env_instance.best_weights
    best_u_std = env_instance.best_u_std
    best_lambda_weight = env_instance.best_lambda_weight

    # Print the best parameters
    print("Best weights:", best_weights)
    print("Best u_std:", best_u_std)
    print("Best lambda_weight:", best_lambda_weight)

