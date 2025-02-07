import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


###############################################################################
# Helper function: compute discrete linearization + LQR around hover
###############################################################################
def compute_lqr_gain_at_hover(params, hover_state, hover_input, dt=0.02):
    """
    1) Linearize continuous dynamics around (x_hover, u_hover).
    2) Discretize (A,B).
    3) Solve discrete LQR for (A_d,B_d,Q_e,R_e) to get feedback gain K.
    """
    import numpy as np
    import casadi as cs
    from scipy.linalg import solve_discrete_are

    Ixx, Iyy, Izz = params['inertia']
    m = params['mass']
    g = params['gravity']

    # Symbolic definition
    x_sym = cs.MX.sym("x", 12)
    u_sym = cs.MX.sym("u", 6)

    # Unpack the state symbols
    px    = x_sym[0]
    py    = x_sym[1]
    pz    = x_sym[2]
    vx    = x_sym[3]
    vy    = x_sym[4]
    vz    = x_sym[5]
    phi   = x_sym[6]
    theta = x_sym[7]
    psi   = x_sym[8]
    p_    = x_sym[9]
    q_    = x_sym[10]
    r_    = x_sym[11]

    # Unpack the controls
    Fx = u_sym[0]
    Fy = u_sym[1]
    Fz = u_sym[2]
    tx = u_sym[3]
    ty = u_sym[4]
    tz = u_sym[5]

    # Same continuous dynamics as before
    px_dot = vx
    py_dot = vy
    pz_dot = vz

    vx_dot = (Fx / m) - g*(cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
    vy_dot = (Fy / m) - g*(cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
    vz_dot = (Fz / m) - g*(cs.cos(phi)*cs.cos(theta))

    phi_dot   = p_ + q_*cs.sin(phi)*cs.tan(theta) + r_*cs.cos(phi)*cs.tan(theta)
    theta_dot = q_*cs.cos(phi) - r_*cs.sin(phi)
    psi_dot   = q_*cs.sin(phi)/cs.cos(theta) + r_*cs.cos(phi)/cs.cos(theta)

    p_dot = (1.0 / Ixx) * (tx + (Iyy - Izz)*q_*r_)
    q_dot = (1.0 / Iyy) * (ty + (Izz - Ixx)*p_*r_)
    r_dot = (1.0 / Izz) * (tz + (Ixx - Iyy)*p_*q_)

    f_expl = cs.vertcat(px_dot, py_dot, pz_dot,
                        vx_dot, vy_dot, vz_dot,
                        phi_dot, theta_dot, psi_dot,
                        p_dot, q_dot, r_dot)

    # 1) Create CasADi expressions for the Jacobians
    A_expr = cs.jacobian(f_expl, x_sym)
    B_expr = cs.jacobian(f_expl, u_sym)

    # 2) Wrap them as CasADi functions
    A_fun = cs.Function("A_fun", [x_sym, u_sym], [A_expr])
    B_fun = cs.Function("B_fun", [x_sym, u_sym], [B_expr])

    # 3) Evaluate at the hover point (hover_state, hover_input)
    A_val = A_fun(hover_state, hover_input)
    B_val = B_fun(hover_state, hover_input)

    # Convert to numpy
    A_val = np.array(A_val).astype(float)
    B_val = np.array(B_val).astype(float)

    # 4) Discretize with forward Euler
    A_d = np.eye(12) + A_val * dt
    B_d = B_val * dt

    # 5) Solve discrete LQR
    Q_e = np.diag([1]*3 + [0.1]*3 + [0.1]*3 + [0.05]*3)
    R_e = np.diag([1e-2]*6)

    P = solve_discrete_are(A_d, B_d, Q_e, R_e)
    K = np.linalg.inv(R_e + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)

    return A_d, B_d, K


###############################################################################
# Tube MPC class
###############################################################################
class TubeMPC:
    def __init__(self, params):
        """
        Build a 'tube MPC' around a nominal trajectory.
        
        For simplicity:
        - We use a single, constant linearization around hover.
        - We compute one LQR feedback gain K offline.
        - We do naive constraint tightening via margins.
        
        In advanced applications, you'd re-linearize and recompute K 
        around your current nominal trajectory, and compute more precise 
        constraint tightening sets (robust invariant sets).
        """
        self.params = params
        self.dt = params["dt"]
        self.horizon = 5  # Prediction horizon in # of steps

        # 1) Build the nominal model (same as your original full MPC model)
        self.model = self._build_nominal_model()

        # 2) Build the nominal ACADOS solver (with tightened constraints)
        self.ocp_solver = self._build_acados_ocp_solver()

        # 3) Pre-compute local linear feedback gain around hover
        x_hover = np.zeros(12)
        u_hover = np.array([0.0, 0.0, params['mass']*params['gravity'], 0.0, 0.0, 0.0])
        _, _, self.K = compute_lqr_gain_at_hover(params, x_hover, u_hover, dt=0.02)

        self.initialized = True

    def _build_nominal_model(self):
        """
        Re-implement your 12D hexarotor dynamics in CasADi for ACADOS.
        """
        Ixx, Iyy, Izz = self.params["inertia"]
        m   = self.params["mass"]
        g   = self.params["gravity"]

        x  = cs.MX.sym("x", 12)
        u  = cs.MX.sym("u", 6)
        xd = cs.MX.sym("xdot", 12)

        # Index into x and u explicitly
        px    = x[0]
        py    = x[1]
        pz    = x[2]
        vx    = x[3]
        vy    = x[4]
        vz    = x[5]
        phi   = x[6]
        theta = x[7]
        psi   = x[8]
        p_    = x[9]
        q_    = x[10]
        r_    = x[11]

        Fx    = u[0]
        Fy    = u[1]
        Fz    = u[2]
        tx    = u[3]
        ty    = u[4]
        tz    = u[5]

        px_dot = vx
        py_dot = vy
        pz_dot = vz
        vx_dot = (Fx / m) - g*(cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
        vy_dot = (Fy / m) - g*(cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
        vz_dot = (Fz / m) - g*(cs.cos(phi)*cs.cos(theta))

        phi_dot   = p_ + q_*cs.sin(phi)*cs.tan(theta) + r_*cs.cos(phi)*cs.tan(theta)
        theta_dot = q_*cs.cos(phi) - r_*cs.sin(phi)
        psi_dot   = q_*cs.sin(phi)/cs.cos(theta) + r_*cs.cos(phi)/cs.cos(theta)

        p_dot = (1.0 / Ixx) * (tx + (Iyy - Izz)*q_*r_)
        q_dot = (1.0 / Iyy) * (ty + (Izz - Ixx)*p_*r_)
        r_dot = (1.0 / Izz) * (tz + (Ixx - Iyy)*p_*q_)

        f_expl = cs.vertcat(px_dot, py_dot, pz_dot,
                            vx_dot, vy_dot, vz_dot,
                            phi_dot, theta_dot, psi_dot,
                            p_dot, q_dot, r_dot)

        f_impl = xd - f_expl

        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xd
        model.u = u
        model.name = "tube_hexarotor_model"

        return model

    def _build_acados_ocp_solver(self):
        """
        Build the nominal OCP with constraint tightening.
        """
        ocp = AcadosOcp()
        ocp.model = self.model

        # Prediction horizon
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon

        # Set the initial condition constraints to be updated at runtime
        ocp.constraints.x0 = np.zeros(12)

        # Cost weights (nominal):
        p = self.params
        Q = np.diag([
            p['tracking_weight_pos'],
            p['tracking_weight_pos'],
            p['tracking_weight_pos']*16,
            p['tracking_weight_vel'],
            p['tracking_weight_vel'],
            p['tracking_weight_vel'],
            p['tracking_weight_att'],
            p['tracking_weight_att'],
            p['tracking_weight_att'],
            p['tracking_weight_ang_vel'],
            p['tracking_weight_ang_vel'],
            p['tracking_weight_ang_vel']
        ])
        R = np.diag([
            p['control_weight'],
            p['control_weight'],
            p['control_weight'],
            p['control_weight']*10,
            p['control_weight']*10,
            p['control_weight']*10
        ])

        # Combine into W
        W = np.block([
            [Q, np.zeros((12, 6))],
            [np.zeros((6, 12)), R]
        ])
        ocp.cost.W = W

        # The residual dimension = 12 state + 6 input = 18
        ocp.cost.Vx = np.eye(18)[:, :12]
        ocp.cost.Vu = np.eye(18)[:, 12:]
        ocp.cost.yref = np.zeros(18)

        # Tightened actuator constraints
        max_force = p['max_force']
        max_torque = p['max_torque']

        # Naive margin to account for possible feedback corrections
        margin_F = 2.0
        margin_T = 0.01

        lbu = np.array([
            -max_force + margin_F,
            -max_force + margin_F,
            0.0,  # let’s assume Fz >= 0
            -max_torque + margin_T,
            -max_torque + margin_T,
            -max_torque + margin_T
        ])
        ubu = np.array([
            max_force - margin_F,
            max_force - margin_F,
            12*max_force,   # or some upper bound on Fz
            max_torque - margin_T,
            max_torque - margin_T,
            max_torque - margin_T
        ])
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.arange(6)

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # Build the solver
        return AcadosOcpSolver(ocp)

    def compute_control(self, x_real, x_nom, u_nom):
        """
        1) Update the nominal OCP with x_nom(0) as the initial state.
        2) Solve the OCP to get u_nom*(0).
        3) Compute the real input: u_real = u_nom*(0) - K*(x_real - x_nom(0)).

        In advanced tube MPC, you'd do:
        - x_nom(k) is the predicted nominal state at time k,
        - Possibly a time-varying K_k, re-linearized at each step.
        """
        if not self.initialized:
            return np.zeros(6)

        # 1) Force the OCP's initial constraint to x_nom
        self.ocp_solver.set(0, "lbx", x_nom)
        self.ocp_solver.set(0, "ubx", x_nom)

        # 2) Set references. We'll track x_nom, keep the input near u_nom.
        yref = np.concatenate([x_nom, u_nom])
        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            # (optional) set initial guess for controls
            self.ocp_solver.set(i, "u", u_nom)

        # Solve
        status = self.ocp_solver.solve()
        if status != 0:
            print("[TubeMPC] ACADOS solver status:", status, 
                  " -- fallback to nominal input or handle infeasibility here.")

        # 3) Extract the nominal control at stage 0
        u_nom0 = np.array(self.ocp_solver.get(0, "u"))

        # 4) Apply local feedback
        e = x_real - x_nom
        u_real = u_nom0 - self.K.dot(e)

        return u_real


###############################################################################
# Example usage
###############################################################################
def main():
    # Example parameters
    params = {
        'inertia': [0.115125971, 0.116524229, 0.230387752],
        'mass': 7.0,
        'gravity': 9.81,
        'max_force': 20.0,
        'max_torque': 0.05,
        'control_weight': 0.005,
        'tracking_weight_pos': 10,
        'tracking_weight_vel': 3,
        'tracking_weight_att': 80,
        'tracking_weight_ang_vel': 50,
        'dt': 0.3
    }

    # Build Tube MPC
    tube_mpc = TubeMPC(params)

    # We'll integrate the real system at a smaller timestep, say dt_sim = 0.01
    dt_sim = 0.01
    steps  = 1000

    # Initial real state
    x_real = np.zeros(12)
    states_log = [x_real.copy()]
    controls_log = []

    # Nominal guess for x_nom, u_nom
    x_nom = x_real.copy()
    u_nom = np.array([0.0, 0.0, params["mass"]*params["gravity"], 0.0, 0.0, 0.0])

    # Build a function for continuous dynamics
    x_sym = tube_mpc.model.x
    u_sym = tube_mpc.model.u
    f_expl = tube_mpc.model.f_expl_expr
    f_func = cs.Function('f_func', [x_sym, u_sym], [f_expl])

    # Suppose we want to hover at pz=2. We'll keep x_nom = [0,0,2,0,0,0,...]
    x_des = x_real.copy()
    x_des[2] = 1.0

    for k in range(steps):
        # For demonstration, keep x_nom constant at x_des.
        x_nom = x_des
        # Also keep u_nom = hover thrust. 
        u_nom = np.array([0.0, 0.0, params["mass"]*params["gravity"], 0.0, 0.0, 0.0])

        # Tube MPC solve
        u_tube = tube_mpc.compute_control(x_real, x_nom, u_nom)

        # Integrate the real system with Euler + mild random disturbance
        disturbance = np.random.randn(12)
        disturbance[:3] = 0.1 * disturbance[:3]
        disturbance[3:6] = 0.01 * disturbance[:3]
        disturbance[6:9] = 0.01 * disturbance[:3]
        disturbance[9:] = 0.001 * disturbance[:3]
        x_dot = np.array(f_func(x_real, u_tube)).flatten()
        x_real = x_real + dt_sim * x_dot + disturbance * dt_sim

        states_log.append(x_real.copy())
        controls_log.append(u_tube)

    states_log = np.array(states_log)
    controls_log = np.array(controls_log)
    t = np.arange(steps+1)*dt_sim

    # Plot results
    plt.figure()
    plt.plot(t, states_log[:,0], label="px")
    plt.plot(t, states_log[:,1], label="py")
    plt.plot(t, states_log[:,2], label="pz")
    plt.title("Position")
    plt.xlabel("time (s)")
    plt.ylabel("pos (m)")
    plt.legend()
    plt.grid(True)

    roll  = states_log[:,6]*180.0/np.pi
    pitch = states_log[:,7]*180.0/np.pi
    yaw   = states_log[:,8]*180.0/np.pi
    plt.figure()
    plt.plot(t, roll,  label="roll (deg)")
    plt.plot(t, pitch, label="pitch (deg)")
    plt.plot(t, yaw,   label="yaw (deg)")
    plt.title("Orientation")
    plt.xlabel("time (s)")
    plt.ylabel("angle (deg)")
    plt.legend()
    plt.grid(True)

    plt.show()
    print("Final state:", states_log[-1])
    print("Final control:", controls_log[-1])

if __name__ == "__main__":
    main()
