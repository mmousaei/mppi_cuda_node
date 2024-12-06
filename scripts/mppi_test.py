import numpy as np
import math
import matplotlib.pyplot as plt
from mppi_numba_gravity import MPPI_Numba, Config


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


    mppi_params = dict(
        # Task specification
        dt=cfg.dt,
        x0=x0, # Start state
        xgoal=xgoal, # Goal position

        # For risk-aware min time planning
        goal_tolerance=0.001,
        dist_weight=2000, #  Weight for dist-to-goal cost.
        # dist_weights = np.array([200, 200, 500, 0, 0, 0, 1000, 1000, 2000, 0, 0, 0]),
        lambda_weight=10, # Temperature param in MPPI
        num_opt=2, # Number of steps in each solve() function call.

        # Control and sample specification
        u_std=np.array([1, 1, 1, 0.005, 0.005, 0.005]), # Noise std for sampling linear and angular velocities.
        vrange = np.array([-60.0, 60.0]), # Linear velocity range.
        wrange=np.array([-0.1, 0.1]), # Angular velocity range.
        # weights = np.array([150, 150, 300, 15, 1500, 1500, 3000, 100, 1, 5, 5, 1, 100]), # w_pose_x, w_pose_y, w_pose_z, w_vel, w_att_roll, w_att_pitch, w_att_yaw, w_omega, w_cont, w_cont_m, w_cont_f, w_cont_M, w_terminal
        weights = np.array([4550, 4550, 5300,       # w_pose_x, w_pose_y, w_pose_z
                            600, 600, 600,          # w_vel_x, w_vel_y, w_vel_z
                            85000, 85000, 85000,   # w_att_roll, w_att_pitch, w_att_yaw
                            500, 500, 500,    # w_omega_x, w_omega_y, w_omega_z
                            1, 5, 5, 1,             # w_cont, w_cont_m, w_cont_f, w_cont_M 
                            1000]),                 # w_terminal
        inertia_mass = np.array([0.115125971, 0.116524229, 0.230387752, 7.00]) # I_xx, I_yy, I_zz, mass
    )

    mppi_controller = MPPI_Numba(cfg)
    mppi_controller.set_params(mppi_params)

    # Loop
    max_steps = 1000
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
        gravity_vector_world = np.array([0, 0, 9.81*7.00])
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                       np.cos(phi)*np.cos(theta)]
        ])
        gravity_body = np.dot(R.T, gravity_vector_world)  # Rotate gravity to body frame
        # u_curr[:3] += gravity_body
        uhist[t] = u_curr

        # Simulate state forward 
        xhist[t+1, :] = dynamics_update_sim(xhist[t, :], u_curr, cfg.dt)
        # print("x: ", xhist[t+1, :])
        print(t)
        # Update MPPI state (x0, useq)
        mppi_controller.shift_and_update(xhist[t+1], useq, num_shifts=1)

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
