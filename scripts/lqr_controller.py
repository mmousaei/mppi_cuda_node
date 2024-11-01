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
        qo = 515
        qw = 1
        qwrw = 0.01
        cf = 1
        cw = 1
        ctrw = 0.01
        self.Q = np.diag([qp, qp, qp*2, qv, qv, qv, qo*4, qo*4, qo*10, qw, qw, qw, qwrw])
        self.R = np.diag([cf, cf, cf, cw, cw, cw, ctrw])
        self.desired_x = np.zeros(13)
        self.desired_x[0] = 1
        self.desired_x[1] = -1
        self.desired_x[2] = 2

        # Hexarotor parameters (changed later)
        self.m = 2  # Mass of the hexarotor 
        self.J = np.diag([0.00562345, 0.00290001, 0.00289597])  # Inertia matrix
        self.J_rw = np.diag([0, 0, 0.01]) # Inertia matrix for the reaction wheel

    def hex_dynamics(self, x, u):
        [p, v, psi, omega], omega_rw = np.split(x[:12], 4), x[12]/148 # will have f_ee and m_ee later between the omegas
        f_T, m_T, t_rw = u[:3], u[3:6], u[6]
        phi, theta, psi = Psi
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]
        ])
        gravity_world = np.array([0, 0, self.g])
        gravity_body = np.dot(R.T, gravity_world)  # Rotate gravity to body frame

        nu = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])

        p_dot = v
        v_dot = (1/self.m) * f_T - gravity_body
        psi_dot = np.dot(nu, omega)
        omega_dot = np.dot(np.linalg.inv(self.J), m_T - np.cross(omega, np.dot(self.J, omega)))

        return np.concatenate([p_dot, v_dot, psi_dot, omega_dot])

    def get_gravity_body(self, x):
        _, _, Psi, _ = np.split(x, 4)
        phi, theta, psi = Psi
        R = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]
        ])
        gravity_world = np.array([0, 0, self.g])
        gravity_body = np.dot(R.T, gravity_world)  # Rotate gravity to body frame
        return gravity_body
    
    def hex_linearized_dynamics(self, x):
        [_, _, Psi, omega], omega_rw = np.split(x[:12], 4), x[12]/148 # will have f_ee and m_ee later between the omegas
        phi, theta, _ = Psi
        g = 0
        # Define the linearized dynamics matrices A and B
        A = np.array([
            [0, 0, 0,           1, 0, 0,            0,                                                                           0,                                                                                                                0,               0,                                                                   0,                                                                                           0,                                                                                                  0],
            [0, 0, 0,           0, 1, 0,            0,                                                                           0,                                                                                                                0,               0,                                                                   0,                                                                                           0,                                                                                                  0],
            [0, 0, 0,           0, 0, 1,            0,                                                                           0,                                                                                                                0,               0,                                                                   0,                                                                                           0,                                                                                                  0],
            [0, 0, 0,           0, 0, 0,            0,                                                                          -g*np.cos(theta),                                                                                                  0,               0,                                                                   0,                                                                                           0,                                                                                                  0],
            [0, 0, 0,           0, 0, 0,            g*np.cos(phi)*np.cos(theta),                                                -g*np.sin(phi)*np.sin(theta),                                                                                      0,               0,                                                                   0,                                                                                           0,                                                                                                  0],
            [0, 0, 0,           0, 0, 0,           -g*np.sin(phi)*np.cos(theta),                                                -g*np.cos(phi)*np.sin(theta),                                                                                      0,               0,                                                                   0,                                                                                           0,                                                                                                  0],
            [0, 0, 0,           0, 0, 0,            omega[1]*np.cos(phi)*np.tan(theta) - omega[2]*np.sin(phi)*np.tan(theta),     omega[1]*(np.tan(theta)**2 + 1)*np.sin(phi) + omega[2]*(np.tan(theta)**2 + 1)*np.cos(phi),                        0,               1,                                                                   np.sin(phi)*np.tan(theta),                                                                   np.cos(phi)*np.tan(theta),                                                                          0],
            [0, 0, 0,           0, 0, 0,            -omega[1]*np.sin(phi) - omega[2]*np.cos(phi),                                0,                                                                                                                0,               0,                                                                   np.cos(phi),                                                                                 -np.sin(phi),                                                                                       0],
            [0, 0, 0,           0, 0, 0,            omega[1]*np.cos(phi)/np.cos(theta) - omega[2]*np.sin(phi)/np.cos(theta),     omega[1]*np.sin(theta)*np.sin(phi)/(np.cos(theta)**2) + omega[2]*np.sin(theta)*np.cos(phi)/(np.cos(theta)**2),    0,               0,                                                                   np.sin(phi)/np.cos(theta),                                                                   np.cos(phi)/np.cos(theta),                                                                          0],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                                0,               0,                                                                   (self.J[1][1]*omega[2] - self.J[2][2]*omega[2])/self.J[0][0],                                (self.J[1][1]*omega[1] - self.J[2][2]*omega[1])/self.J[0][0],                                       0],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                                0,               (-self.J[0][0]*omega[2] + self.J[2][2]*omega[2])/self.J[1][1],       0,                                                                                           self.J_rw[2][2]*omega_rw + (-self.J[0][0]*omega[0] + self.J[2][2]*omega[0])/self.J[1][1],           self.J_rw[2][2]*omega[2]],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                                0,               (-self.J[0][0]*omega[1] + self.J[1][1]*omega[1])/self.J[2][2],       -self.J_rw[2][2]*omega_rw + (-self.J[0][0]*omega[0] + self.J[1][1]*omega[0])/self.J[2][2],   0,                                                                                                  -self.J_rw[2][2]*omega[1]],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                                0,               0,                                                                   0,                                                                                           0,                                                                                                  0]
        ])

        B = np.array([
            [0,        0,        0,        0,              0,              0,              0],
            [0,        0,        0,        0,              0,              0,              0],
            [0,        0,        0,        0,              0,              0,              0],
            [1/self.m, 0,        0,        0,              0,              0,              0],
            [0,        1/self.m, 0,        0,              0,              0,              0],
            [0,        0,        1/self.m, 0,              0,              0,              0],
            [0,        0,        0,        0,              0,              0,              0],
            [0,        0,        0,        0,              0,              0,              0],
            [0,        0,        0,        0,              0,              0,              0],
            [0,        0,        0,        1/self.J[0][0], 0,              0,              0],
            [0,        0,        0,        0,              1/self.J[1][1], 0,              0],
            [0,        0,        0,        0,              0,              1/self.J[2][2], 0],
            [0,        0,        0,        0,              0,              0,              1/self.J_rw[2][2]]
        ])

        return A, B
    
    def lqr_control(self, x):
        A, B = self.hex_linearized_dynamics(x)
        try:
            # Compute the LQR gain matrix K
            K, _, _ = ctrl.lqr(A, B, self.Q, self.R)
        except Exception as e:
            print("Error computing LQR gain matrix:", e)
            K = np.zeros((6, 13))  # Fallback to a zero matrix if LQR fails

        # Compute the error state
        e = x - self.desired_x
        
        # LQR control law applied to the error state
        u = -np.dot(K, e)
        return u

if __name__ == '__main__':
    controller = LqrController()
    x = np.zeros(12)  # Initial state
    history = []

    for i in range(500):
        print(i)
        u = controller.lqr_control(x.copy())
        gravity_vector_body = controller.get_gravity_body(x.copy())
        u[:3] += gravity_vector_body
        x_dot = controller.hex_dynamics(x.copy(), u.copy())
        x += x_dot * controller.dt
        history.append(x.copy())

    history = np.array(history)

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.plot(history[:, i], label=f'Position {i}')
    for i in range(3, 6):
        plt.plot(history[:, i], label=f'Velocity {i-3}')
    for i in range(6, 9):
        plt.plot(history[:, i], label=f'Euler Angle {i-6}')
    for i in range(9, 12):
        plt.plot(history[:, i], label=f'Angular Velocity {i-9}')
    
    plt.xlabel('Time step')
    plt.ylabel('State values')
    plt.legend()
    plt.show()
