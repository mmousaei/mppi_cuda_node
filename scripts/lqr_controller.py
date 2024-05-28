import numpy as np
import control as ctrl

class LqrController:
    def __init__(self):
        self.dt = 0.02
        self.g = 9.81
        # Initialize LQR parameters
        qp = 20
        qv = 1
        qo = 515
        qw = 1
        cf = 1
        cw = 1
        self.Q = np.diag([qp, qp, qp*2, qv, qv, qv, qo*4, qo*4, qo*10, qw, qw, qw])  # Adjust these values
        self.R = np.diag([cf ,cf, cf, cw, cw, cw])  # Adjust these values
        self.desired_x = np.zeros(12)  # Desired state [position, velocity, euler angle, angular velocity]
        self.desired_x[0] = 1
        self.desired_x[1] = -1
        self.desired_x[2] = 2
        # self.desired_x[6] = np.deg2rad(20)
        # self.desired_x[7] = np.deg2rad(-10)
        # self.desired_x[8] = np.deg2rad(30)

        # Hexarotor parameters
        self.m = 2 # Mass of the hexarotor 
        self.J = np.diag([0.00562345, 0.00290001, 0.00289597])  # Inertia matrix

    def hex_dynamics(self, x, u):
        
        p, v, psi, omega = np.split(x, 4)
        f_T, m_T = u[:3], u[3:]
        # if x[0] > 12.6:
        #     f_T[0] += f_ee
        phi, theta, _ = psi
        R = np.array([
            [np.cos(theta)*np.cos(phi), np.sin(phi)*np.sin(theta)*np.cos(phi) - np.cos(phi)*np.sin(phi), np.cos(phi)*np.sin(theta)*np.cos(phi) + np.sin(phi)*np.sin(phi)],
            [np.cos(theta)*np.sin(phi), np.sin(phi)*np.sin(theta)*np.sin(phi) + np.cos(phi)*np.cos(phi), np.cos(phi)*np.sin(theta)*np.sin(phi) - np.sin(phi)*np.cos(phi)],
            [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]
        ])
        gravity_world = np.array([0, 0, 9.81])
        gravity_body = np.dot(R.T, gravity_world)  # Rotate gravity to body frame

        p_dot = v
        v_dot = (1/self.m) * f_T - gravity_body
        psi_dot = np.dot(R, omega)
        omega_dot = np.dot(np.linalg.inv(self.J), m_T - np.cross(omega, np.dot(self.J, omega)))

        return np.concatenate([p_dot, v_dot, psi_dot, omega_dot])
    
    def hex_linearized_dynamics(self, x):
        _, _, psi, omega = np.split(x, 4)
        phi, theta, _ = psi
        g = 0
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
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               0,                                                                   (self.J[1][1]*omega[2] - self.J[2][2]*omega[2])/self.J[0][0],       (self.J[1][1]*omega[1] - self.J[2][2]*omega[1])/self.J[0][0]],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               (-self.J[0][0]*omega[2] + self.J[2][2]*omega[2])/self.J[1][1],       0,                                                                  (-self.J[0][0]*omega[0] + self.J[2][2]*omega[0])/self.J[1][1]],
            [0, 0, 0,           0, 0, 0,            0,                                                                           0,                                                                                                            0,               (self.J[0][0]*omega[1] - self.J[1][1]*omega[1])/self.J[2][2],        (self.J[0][0]*omega[0] - self.J[1][1]*omega[0])/self.J[2][2],       0]
        ])

        B = np.array([
            [0,        0,        0,        0,              0,              0],
            [0,        0,        0,        0,              0,              0],
            [0,        0,        0,        0,              0,              0],
            [1/self.m, 0,        0,        0,              0,              0],
            [0,        1/self.m, 0,        0,              0,              0],
            [0,        0,        1/self.m, 0,              0,              0],
            [0,        0,        0,        0,              0,              0],
            [0,        0,        0,        0,              0,              0],
            [0,        0,        0,        0,              0,              0],
            [0,        0,        0,        1/self.J[0][0], 0,              0],
            [0,        0,        0,        0,              1/self.J[1][1], 0],
            [0,        0,        0,        0,              0,              1/self.J[2][2]]
        ])

        return A, B
    
    def lqr_control(self, x):
        
        A, B = self.hex_linearized_dynamics(x)
        # Compute the LQR gain matrix K
        K, _, _ = ctrl.lqr(A, B, self.Q, self.R)

        # Compute the error state
        e = x - self.desired_x
        
        # LQR control law applied to the error state
        u = -np.dot(K, e)
        return u