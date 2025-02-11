import numpy as np
import math
import matplotlib.pyplot as plt
from mppi_numba_gravity import MPPI_Numba, Config
from scipy.signal import butter

def butter_lowpass_online(cutoff, fs, order=1):
    """
    Design a low-pass Butterworth filter and return coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

class OnlineLPF:
    def __init__(self, b, a, num_controls):
        """
        Initialize the LPF with given coefficients and number of controls.
        """
        self.b = b
        self.a = a
        self.prev_input = np.zeros(num_controls)
        self.prev_output = np.zeros(num_controls)

    def filter(self, u_curr):
        """
        Apply the LPF to the current control inputs.
        """
        # Calculate filtered control: y[n] = b[0]*x[n] + b[1]*x[n-1] - a[1]*y[n-1]
        filtered_u = (
            self.b[0] * u_curr +
            self.b[1] * self.prev_input -
            self.a[1] * self.prev_output
        )

        # Update filter states
        self.prev_input = u_curr
        self.prev_output = filtered_u

        return filtered_u
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

if __name__ == "__main__":
    num_controls = 6
    num_states = 12
    cfg = Config(T = 0.6,
            dt = 0.02,
            num_control_rollouts = 1024,#int(2e4), # Same as number of blocks, can be more than 1024
            num_controls = num_controls,
            num_states = num_states,
            num_vis_state_rollouts = 1,
            seed = 1)
    # x0 = np.array([2,-1, 3, 0, 0, 0, 0.1, -0.1, 0.3, 0, 0, 0])
    x0 = np.zeros(12)
    xgoal = np.array([2,-1, 3, 0, 0, 0, 0.1, -0.1, 0.3, 0, 0, 0])
    # xgoal = np.array([2,-1, 3, 0, 0, 0, 0.0, -0.0, 0.0, 0, 0, 0])


    mppi_params = dict(
        # Task specification
        dt=cfg.dt,
        x0=x0, # Start state
        xgoal=xgoal, # Goal position

        # For risk-aware min time planning
        goal_tolerance=0.001,
        dist_weight=2000, #  Weight for dist-to-goal cost.
        # dist_weights = np.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0]),
        lambda_weight=17.782301664352417, # Temperature param in MPPI
        num_opt=2, # Number of steps in each solve() function call.

        # Control and sample specification
        u_std=np.array([0.4       , 0.51262367, 0.48734699, 0.006     , 0.004     ,
       0.00479975]), # Noise std for sampling linear and angular velocities.
        vrange = np.array([-60.0, 60.0]), # Linear velocity range.
        wrange=np.array([-0.1, 0.1]), # Angular velocity range.
        # weights = np.array([150, 150, 300, 15, 1500, 1500, 3000, 100, 1, 5, 5, 1, 100]), # w_pose_x, w_pose_y, w_pose_z, w_vel, w_att_roll, w_att_pitch, w_att_yaw, w_omega, w_cont, w_cont_m, w_cont_f, w_cont_M, w_terminal
        weights = np.array([1.54400000e+04, 2.55382091e+04, 1.95739902e+04, 1.80000000e+03,
       1.80000000e+03, 1.20000000e+03, 1.00000000e+05, 1.00000000e+05,
       1.00000000e+05, 2.57381921e+04, 3.36000000e+04, 1.52413766e+04,
       1.20000000e+00, 1.18098271e+03, 8.00000000e-01, 9.85520786e+02,
       5.66759663e+04]),                 # w_terminal
        inertia_mass = np.array([0.115125971, 0.116524229, 0.230387752, 7.00]) # I_xx, I_yy, I_zz, mass
    )

    mppi_controller = MPPI_Numba(cfg)
    mppi_controller.set_params(mppi_params)
    

    # Define the LPF parameters
    cutoff_freq = 10  # Cutoff frequency in Hz
    sampling_rate = 1 / cfg.dt  # Sampling rate based on the timestep
    b, a = butter_lowpass_online(cutoff_freq, sampling_rate)
    lpf = OnlineLPF(b, a, num_controls)

    # Loop
    max_steps = 1000
    xhist = np.zeros((max_steps+1, num_states))*np.nan
    uhist = np.zeros((max_steps, num_controls))*np.nan
    xhist[0] = x0

    vis_xlim = [-1, 8]
    vis_ylim = [-1, 6]

    init_control_seq = np.zeros((int(cfg.T/cfg.dt), cfg.num_controls))
    # mppi_controller.shift_and_update(x0, init_control_seq, num_shifts=1)

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
        # Apply the LPF online to the current control commands
        filtered_u_curr = lpf.filter(u_curr)

        # Simulate state forward using filtered controls
        xhist[t + 1, :] = dynamics_update_sim(xhist[t, :], filtered_u_curr, cfg.dt)
        # xhist[t + 1, :] = dynamics_update_sim(xhist[t, :], u_curr, cfg.dt)
        # u_curr[:3] += gravity_body
        uhist[t] = filtered_u_curr
        # uhist[t] = u_curr

        # Simulate state forward 
        xhist[t+1, :] = dynamics_update_sim(xhist[t, :], u_curr, cfg.dt)
        # print("x: ", xhist[t+1, :])
        print(t)
        # Update MPPI state (x0, useq)
        mppi_controller.shift_and_update(xhist[t+1], useq, num_shifts=1)

    # Assuming xgoal is your goal position and it has appropriate values for each state
    x_goal, y_goal, z_goal = xgoal[:3]
    roll_goal, pitch_goal, yaw_goal = xgoal[6:9]

    fig, axs = plt.subplots(5, 3, figsize=(12, 9/4*5))  # Create 5 rows and 3 columns

    # -------- First four rows (your original plots) --------
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
    axs[1][1].set_title('Pitch')
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

    # -------- Fifth row: Display MPPI Parameters as Text --------
    # Turn off the axes for all three subplots in the fifth row
    for j in range(3):
        axs[4][j].axis('off')

    # Format the MPPI parameters into a string
    
    param_text = "MPPI Parameters:\n"
    for i, value in enumerate(mppi_params['weights']):
        param_text += f"Weight {i}: {value} | "
        if not (i+1) % 3:
            param_text += "\n"
    param_text += "\n"
    for i, value in enumerate(mppi_params['u_std']):
        param_text += f"Std Dev {i}: {value} | "
        if not (i+1) % 3:
            param_text += "\n"
    
    param_text += f"num_opt: {mppi_params['num_opt']}'" 
    fig.text(0.04, 0.15, param_text, fontsize=10, verticalalignment='top')

    # Place the text into the left subplot of the fifth row
    # axs[4][:].text(0.0, 1.0, param_text, fontsize=10, verticalalignment='top', transform=axs[4][0].transAxes)

    plt.tight_layout()
    plt.show()

