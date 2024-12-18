import numpy as np
import math
import gym
from gym import spaces
import matplotlib.pyplot as plt
from mppi_numba_gravity import MPPI_Numba, Config
from scipy.signal import butter
from copy import deepcopy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def butter_lowpass_online(cutoff, fs, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

class OnlineLPF:
    def __init__(self, b, a, num_controls):
        self.b = b
        self.a = a
        self.prev_input = np.zeros(num_controls)
        self.prev_output = np.zeros(num_controls)

    def filter(self, u_curr):
        filtered_u = (
            self.b[0] * u_curr +
            self.b[1] * self.prev_input -
            self.a[1] * self.prev_output
        )
        self.prev_input = u_curr
        self.prev_output = filtered_u
        return filtered_u

def dynamics_update_sim(x, u, dt):
    I_xx = 0.115125971
    I_yy = 0.116524229
    I_zz = 0.230387752
    mass = 7.00
    g = 9.81

    x_next = x.copy()
    epsilon = 1e-6

    x_next[0] += dt * x[3] 
    x_next[1] += dt * x[4]
    x_next[2] += dt * x[5]

    x_next[3] += dt * ((1/mass) * u[0] - g * (np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) + np.sin(x[6]) * np.sin(x[8])) )
    x_next[4] += dt * ((1/mass) * u[1] - g * (np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) - np.sin(x[6]) * np.cos(x[8])) )
    x_next[5] += dt * ((1/mass) * u[2] - g * (np.cos(x[6]) * np.cos(x[7])) )

    # Avoid division by zero in angle updates
    if abs(np.cos(x[7])) < epsilon:
        x[7] = np.sign(x[7])*(np.pi/2 - 0.001)

    x_next[6] += dt*(x[9] + x[10]*(math.sin(x[6])*math.tan(x[7])) + x[11]*(math.cos(x[6])*math.tan(x[7])))
    x_next[7] += dt*( x[10]*math.cos(x[6]) - x[11]*math.sin(x[6]))
    x_next[8] += dt*( x[10]*math.sin(x[6])/math.cos(x[7]) + x[11]*math.cos(x[6])/math.cos(x[7]))

    x_next[9]  += dt*((1/I_xx) * (u[3] + I_yy * x[10] * x[11] - I_zz * x[10] * x[11]))
    x_next[10] += dt*((1/I_yy) * (u[4] - I_xx * x[9] * x[11] + I_zz * x[9] * x[11]))
    x_next[11] += dt*((1/I_zz) * (u[5] + I_xx * x[9] * x[10] - I_yy * x[9] * x[10]))

    # Check for NaN/Inf
    if np.any(np.isnan(x_next)) or np.any(np.isinf(x_next)):
        x_next = np.nan_to_num(x_next)

    return x_next

class MPPIEnvironmentWrapper(gym.Env):
    """
    A Gym wrapper for the MPPI tuning. Each episode:
    - The agent observes current parameters (param_vector).
    - Takes one action (increments each parameter by action[i]*(param_vector[i]/100)).
    - We run a full simulation with those updated parameters.
    - Compute reward.
    - Episode ends (done=True).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MPPIEnvironmentWrapper, self).__init__()
        self.num_controls = 6
        self.num_states = 12
        self.cfg = Config(
            T=0.6,
            dt=0.02,
            num_control_rollouts=1024,
            num_controls=self.num_controls,
            num_states=self.num_states,
            num_vis_state_rollouts=1,
            seed=1
        )

        self.x0 = np.zeros(12)
        self.xgoal = np.array([2, -1, 3, 0, 0, 0, 0.1, -0.1, 0.3, 0, 0, 0])

        default_weights = np.array([19300, 22300, 19300,
                                    1500, 1500, 1500,
                                    98500, 98500, 98500,
                                    28000, 28000, 18000,
                                    1, 1000, 1, 1000,
                                    60000])

        self.default_params = dict(
            dt=self.cfg.dt,
            x0=self.x0,
            xgoal=self.xgoal,
            goal_tolerance=0.001,
            dist_weight=2000,
            lambda_weight=20,
            num_opt=2,
            u_std=np.array([0.5, 0.5, 0.5, 0.005, 0.005, 0.005]),
            vrange = np.array([-60.0, 60.0]),
            wrange=np.array([-0.1, 0.1]),
            weights = default_weights,
            inertia_mass = np.array([0.115125971, 0.116524229, 0.230387752, 7.00])
        )

        self.mppi_controller = MPPI_Numba(self.cfg)
        self.mppi_controller.set_params(self.default_params)

        cutoff_freq = 10
        sampling_rate = 1 / self.cfg.dt
        b, a = butter_lowpass_online(cutoff_freq, sampling_rate)
        self.lpf = OnlineLPF(b, a, self.num_controls)

        # Parameter vector layout:
        # [lambda_weight(1), u_std(6), weights(17)] = 24 total
        self.param_size = 1 + 6 + 17
        self.param_indices = {
            'lambda_weight': [0],
            'u_std': list(range(1, 1+6)),
            'weights': list(range(7, 7+17))
        }

        obs_low = np.zeros(self.param_size)
        obs_high = np.ones(self.param_size)*1e6
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.param_size,), dtype=np.float32)

        self.param_vector = self._get_default_param_vector()

        # Track best reward and params
        self.best_reward = -np.inf
        self.best_params = None

    def _get_default_param_vector(self):
        p = np.concatenate((
            np.array([self.default_params['lambda_weight']]),
            self.default_params['u_std'],
            self.default_params['weights']
        ))
        return p

    def reset(self):
        self.param_vector = self._get_default_param_vector()
        return self._get_obs()

    def _get_obs(self):
        return self.param_vector.astype(np.float32)

    def step(self, action):
        # Increment parameters by action[i]*(param_vector[i]/10)
        increment = (self.param_vector / 5.0) * action
        self.param_vector += increment

        # Clip parameters
        lw_idx = self.param_indices['lambda_weight'][0]
        self.param_vector[lw_idx] = np.clip(self.param_vector[lw_idx], 1.0, 100.0)

        u_std_idx = self.param_indices['u_std']
        self.param_vector[u_std_idx] = np.clip(self.param_vector[u_std_idx], 0.0, 5.0)

        w_idx = self.param_indices['weights']
        self.param_vector[w_idx] = np.clip(self.param_vector[w_idx], 0.0, 1e5)

        # Update MPPI params
        updated_params = deepcopy(self.default_params)
        updated_params['lambda_weight'] = self.param_vector[lw_idx]
        updated_params['u_std'] = self.param_vector[u_std_idx]
        updated_params['weights'] = self.param_vector[w_idx]
        self.mppi_controller.set_params(updated_params)

        xhist, uhist = self.run_episode(500)
        reward = self.compute_reward(xhist, uhist)

        # Check if this reward is the best so far
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = deepcopy(updated_params)

        print(f"Reward: {reward}\nParameters: {updated_params}")

        obs = self._get_obs()
        done = True
        info = {}
        return obs, reward, done, info

    def run_episode(self, max_steps=500):
        xhist = np.zeros((max_steps+1, self.num_states))*np.nan
        uhist = np.zeros((max_steps, self.num_controls))*np.nan
        xhist[0] = self.x0

        init_control_seq = np.zeros((int(self.cfg.T/self.cfg.dt), self.cfg.num_controls))
        self.mppi_controller.shift_and_update(self.x0, init_control_seq, num_shifts=1)
        for t in range(max_steps):
            useq = self.mppi_controller.solve()
            u_curr = useq[0]

            filtered_u_curr = self.lpf.filter(u_curr)
            filtered_u_curr = np.nan_to_num(filtered_u_curr)
            xhist[t+1, :] = dynamics_update_sim(xhist[t, :], filtered_u_curr, self.cfg.dt)
            uhist[t] = filtered_u_curr
            self.mppi_controller.shift_and_update(xhist[t+1], useq, num_shifts=1)

            if np.any(np.isnan(xhist[t+1])) or np.any(np.isinf(xhist[t+1])):
                break

        return xhist, uhist

    def compute_reward(self, xhist, uhist):
        final_pos = xhist[-1, 0:3]
        goal_pos = self.xgoal[0:3]
        dist = np.linalg.norm(final_pos - goal_pos)

        # Smoothness penalty
        smoothness_penalty = 0.0
        for i in range(1, len(uhist)):
            diff = uhist[i] - uhist[i-1]
            smoothness_penalty += np.sum(diff*diff)

        final_att = xhist[-1, 6:9]
        goal_att = self.xgoal[6:9]
        att_error = np.linalg.norm(final_att - goal_att)

        dist = min(dist, 1e3)
        smoothness_penalty = min(smoothness_penalty, 1e6)
        att_error = min(att_error, np.pi)

        reward = - (dist * 10.0) - (smoothness_penalty * 1000.0) - (att_error * 5000.0)

        if np.isnan(reward) or np.isinf(reward):
            reward += -1e12

        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    # Create the environment
    env = MPPIEnvironmentWrapper()
    vec_env = DummyVecEnv([lambda: env])

    # Create PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Train the model
    model.learn(total_timesteps=300000)

    # After training, print best reward and params
    print("Best Reward Achieved:", env.best_reward)
    print("Parameters for Best Reward:", env.best_params)

    # Test the model
    obs = vec_env.reset()
    for _ in range(5):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print("Test Step Reward:", reward, "Done:", done)
        if done:
            obs = vec_env.reset()
