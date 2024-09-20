import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

class LqrController:
    def __init__(self):
        self.dt = 0.02
        self.g = 9.81
        # Initialize LQR parameters
        qp = 20
        qv = 1
        qo = 20
        qw = 1
        cf = 1
        cw = 1
        self.Q = np.diag([qp, qp, qp*2, qv, qv, qv, qo, qo, qo*2, qw, qw, qw])
        self.R = np.diag([cf, cf, cf, cw, cw, cw])
        self.desired_x = np.zeros(12)
        self.desired_x[0] = 1
        self.desired_x[1] = -1
        self.desired_x[2] = 2

        # Hexarotor parameters
        self.m = 2  # Mass of the hexarotor 
        self.J = np.diag([0.00562345, 0.00290001, 0.00289597])  # Inertia matrix

    def hex_dynamics(self, x, u):
        p, v, Psi, omega = np.split(x, 4)
        f_T, m_T = u[:3], u[3:]
        phi, theta, psi = Psi

        # Rotation matrix from body frame to inertial frame
        R = np.array([
            [np.cos(theta)*np.cos(psi), 
             np.cos(theta)*np.sin(psi), 
             -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), 
             np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), 
             np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), 
             np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), 
             np.cos(phi)*np.cos(theta)]
        ])

        # Corrected gravity vector (points downwards in inertial frame)
        gravity_world = np.array([0, 0, -self.g])
        gravity_body = np.dot(R.T, gravity_world)  # Rotate gravity to body frame

        nu = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])

        p_dot = v
        v_dot = (1/self.m) * f_T + gravity_body  # Add gravity body vector
        psi_dot = np.dot(nu, omega)
        omega_dot = np.dot(np.linalg.inv(self.J), m_T - np.cross(omega, np.dot(self.J, omega)))

        return np.concatenate([p_dot, v_dot, psi_dot, omega_dot])

    def get_gravity_body(self, x):
        _, _, Psi, _ = np.split(x, 4)
        phi, theta, psi = Psi
        R = np.array([
            [np.cos(theta)*np.cos(psi), 
             np.cos(theta)*np.sin(psi), 
             -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), 
             np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), 
             np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), 
             np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), 
             np.cos(phi)*np.cos(theta)]
        ])
        # gravity_world = np.array([0, 0, -self.g])  # Corrected gravity direction
        gravity_world = np.array([0, 0, -0.5538])  # Corrected gravity direction
        gravity_body = np.dot(R.T, gravity_world)  # Rotate gravity to body frame
        return gravity_body

    def hex_linearized_dynamics(self, x):
        # Linearize dynamics around the current state
        _, _, Psi, omega = np.split(x, 4)
        phi, theta, psi = Psi
        omega_x, omega_y, omega_z = omega
        g = self.g

        s_phi = np.sin(phi)
        c_phi = np.cos(phi)
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        s_psi = np.sin(psi)
        c_psi = np.cos(psi)

        A = np.zeros((12, 12))
        B = np.zeros((12, 6))
        # Define the linearized dynamics matrices A and B
        A = np.array([
            [0, 0, 0,           1, 0, 0,            0,                                                                           0,                                                                                                            0,               0,                                                                   0,                                                                  0],
            [0, 0, 0,           0, 1, 0,            0,                                                                           0,                                                                                                            0,               0,                                                                   0,                                                                  0],
            [0, 0, 0,           0, 0, 1,            0,                                                                           0,                                                                                                            0,               0,                                                                   0,                                                                  0],     
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               0,                                                                   0,                                                                  0],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               0,                                                                   0,                                                                  0],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               0,                                                                   0,                                                                  0],
            [0, 0, 0,           0, 0, 0,            omega[1]*np.cos(phi)*np.tan(theta) - omega[2]*np.sin(phi)*np.tan(theta),     omega[1]*(np.tan(theta)**2 + 1)*np.sin(phi) + omega[2]*(np.tan(theta)**2 + 1)*np.cos(phi),                    0,               1,                                                                   np.sin(phi)*np.tan(theta),                                          np.cos(phi)*np.tan(theta)],
            [0, 0, 0,           0, 0, 0,            -omega[1]*np.sin(phi) - omega[2]*np.cos(phi),                                0,                                                                                                            0,               0,                                                                   np.cos(phi),                                                        -np.sin(phi)],
            [0, 0, 0,           0, 0, 0,            omega[1]*np.cos(phi)/np.cos(theta) - omega[2]*np.sin(phi)/np.cos(theta),     omega[1]*np.sin(theta)*np.sin(phi)/np.cos(theta)**2 + omega[2]*np.sin(theta)*np.cos(phi)/np.cos(theta)**2,    0,               0,                                                                   np.sin(phi)/np.cos(theta),                                          np.cos(phi)/np.cos(theta)],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               0,                                                                   (self.J[1,1]*omega[2] - self.J[2,2]*omega[2])/self.J[0,0],       (self.J[1,1]*omega[1] - self.J[2,2]*omega[1])/self.J[0,0]],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               (-self.J[0,0]*omega[2] + self.J[2,2]*omega[2])/self.J[1,1],       0,                                                                  (-self.J[0,0]*omega[0] + self.J[2,2]*omega[0])/self.J[1,1]],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               (self.J[0,0]*omega[1] - self.J[1,1]*omega[1])/self.J[2,2],        (self.J[0,0]*omega[0] - self.J[1,1]*omega[0])/self.J[2,2],       0]
        ])
        # Position dynamics
        A[0, 3] = 1  # dp_x/dt = v_x
        A[1, 4] = 1  # dp_y/dt = v_y
        A[2, 5] = 1  # dp_z/dt = v_z

        # Velocity dynamics (including gravity coupling)
        # v_dot_x
        A[3, 7] = g * c_theta
        # v_dot_y
        A[4, 6] = -g * c_phi * c_theta
        A[4, 7] = g * s_phi * s_theta
        # v_dot_z
        A[5, 6] = g * s_phi * c_theta
        A[5, 7] = g * c_phi * s_theta

        Jx, Jy, Jz = self.J[0, 0], self.J[1, 1], self.J[2, 2]
        # Control inputs to acceleration
        B[3, 0] = 1 / self.m  # f_x affects dv_x/dt
        B[4, 1] = 1 / self.m  # f_y affects dv_y/dt
        B[5, 2] = 1 / self.m  # f_z affects dv_z/dt

        # Control inputs to angular acceleration
        B[9, 3] = 1 / Jx  # m_T_x affects domega_x/dt
        B[10, 4] = 1 / Jy  # m_T_y affects domega_y/dt
        B[11, 5] = 1 / Jz  # m_T_z affects domega_z/dt

        return A, B

    def lqr_control(self, x):
        A, B = self.hex_linearized_dynamics(x)
        try:
            # Compute the LQR gain matrix K
            K, _, _ = ctrl.lqr(A, B, self.Q, self.R)
        except Exception as e:
            print("Error computing LQR gain matrix:", e)
            K = np.zeros((6, 12))  # Fallback to a zero matrix if LQR fails

        # Compute the error state
        e = x - self.desired_x

        # Equilibrium control input to balance gravity
        u_eq = np.zeros(6)
        u_eq[2] = self.m * self.g  # Equilibrium thrust to balance gravity

        # LQR control law applied to the error state
        u = u_eq - np.dot(K, e)
        return u

if __name__ == '__main__':
    controller = LqrController()
    x = np.zeros(12)  # Initial state
    history = []

    for i in range(500):
        u = controller.lqr_control(x.copy())
        x_dot = controller.hex_dynamics(x.copy(), u.copy())
        x += x_dot * controller.dt
        history.append(x.copy())

    history = np.array(history)

    plt.figure(figsize=(12, 8))
    time_steps = np.arange(len(history)) * controller.dt

    plt.subplot(3, 1, 1)
    for i in range(3):
        plt.plot(time_steps, history[:, i], label=f'Position {["x", "y", "z"][i]}')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    for i in range(3, 6):
        plt.plot(time_steps, history[:, i], label=f'Velocity {["x", "y", "z"][i-3]}')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    for i in range(6, 9):
        plt.plot(time_steps, history[:, i], label=f'Euler Angle {["phi", "theta", "psi"][i-6]}')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
