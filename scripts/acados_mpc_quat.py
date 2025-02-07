import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


class OneStepMPC:
    def __init__(self, params):
        """
        Initialize the one-step MPC controller using a quaternion-based hexarotor model.

        Parameters:
        -----------
        params: Dictionary containing system parameters such as inertia, mass, gravity, etc.
        """
        self.params = params
        self.dt = params["dt"]
        self.previous_control = np.zeros(6)  # can be used for smoothness if desired

        # Build the full hexarotor model (quaternion-based)
        self.model = self.build_full_hexarotor_model()

        # Set up horizon
        self.horizon = 5  # e.g., 5 steps over the horizon
        self.ocp_solver = self.build_acados_ocp_solver()
        self.initialized = True

    def build_full_hexarotor_model(self):
        """
        Build a 6-DoF hexarotor model in CasADi using quaternions for attitude representation.

        The state vector is:
          x = [ px, py, pz, vx, vy, vz, q0, q1, q2, q3, ωx, ωy, ωz ]
        where q = [q0, q1, q2, q3] is a unit quaternion (q0 is the scalar part).

        The control vector is:
          u = [ Fx, Fy, Fz, τx, τy, τz ]
        """
        # Extract parameters
        I_xx, I_yy, I_zz = self.params["inertia"]   # moments of inertia
        mass = self.params["mass"]
        g    = self.params["gravity"]

        # Define CasADi symbolic variables
        x = cs.MX.sym("x", 13)      # 13D state
        u = cs.MX.sym("u", 6)       # 6D control
        xdot = cs.MX.sym("xdot", 13)  # time derivative of state

        # ----- State extraction -----
        # Position and velocity (world frame)
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        # Quaternion (attitude), with q0 = scalar part
        q0, q1, q2, q3 = x[6], x[7], x[8], x[9]
        # Angular velocity (body frame)
        wx, wy, wz = x[10], x[11], x[12]

        # ----- Control extraction -----
        Fx, Fy, Fz = u[0], u[1], u[2]
        tau_x, tau_y, tau_z = u[3], u[4], u[5]

        # ----- Position derivatives -----
        px_dot = vx
        py_dot = vy
        pz_dot = vz

        # ----- Velocity derivatives -----
        # Compute rotation matrix R(q) from body frame to world frame (using q = [q0, q1, q2, q3])
        R11 = 1 - 2*(q2**2 + q3**2)
        R12 = 2*(q1*q2 - q0*q3)
        R13 = 2*(q1*q3 + q0*q2)
        R21 = 2*(q1*q2 + q0*q3)
        R22 = 1 - 2*(q1**2 + q3**2)
        R23 = 2*(q2*q3 - q0*q1)
        R31 = 2*(q1*q3 - q0*q2)
        R32 = 2*(q2*q3 + q0*q1)
        R33 = 1 - 2*(q1**2 + q2**2)

        # Compute acceleration: note that gravity is [0, 0, -g] in world coordinates.
        ax = (1 / mass) * (R11 * Fx + R12 * Fy + R13 * Fz)
        ay = (1 / mass) * (R21 * Fx + R22 * Fy + R23 * Fz)
        az = (1 / mass) * (R31 * Fx + R32 * Fy + R33 * Fz) - g

        vx_dot = ax
        vy_dot = ay
        vz_dot = az

        # ----- Quaternion kinematics -----
        # q_dot = 1/2 * quatmultiply(q, [0, ω]) 
        q0_dot = -0.5*(q1*wx + q2*wy + q3*wz)
        q1_dot = 0.5*(q0*wx + q2*wz - q3*wy)
        q2_dot = 0.5*(q0*wy - q1*wz + q3*wx)
        q3_dot = 0.5*(q0*wz + q1*wy - q2*wx)

        # ----- Angular velocity dynamics -----
        wx_dot = (1.0 / I_xx) * (tau_x + (I_yy - I_zz) * wy * wz)
        wy_dot = (1.0 / I_yy) * (tau_y + (I_zz - I_xx) * wx * wz)
        wz_dot = (1.0 / I_zz) * (tau_z + (I_xx - I_yy) * wx * wy)

        # ----- Collect explicit dynamics -----
        f_expl = cs.vertcat(
            px_dot,
            py_dot,
            pz_dot,
            vx_dot,
            vy_dot,
            vz_dot,
            q0_dot,
            q1_dot,
            q2_dot,
            q3_dot,
            wx_dot,
            wy_dot,
            wz_dot
        )

        # Implicit form: xdot - f_expl = 0
        f_impl = xdot - f_expl

        # Build the AcadosModel
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "full_hexarotor_model_quat"

        return model

    def build_acados_ocp_solver(self):
        """
        Build the ACADOS OCP solver for the quaternion-based hexarotor.

        The cost tracks state error and control usage. Note that the state now is 13-dimensional:
          - position: indices 0-2
          - velocity: indices 3-5
          - quaternion: indices 6-9
          - angular velocity: indices 10-12
        """
        params = self.params
        ocp = AcadosOcp()
        ocp.model = self.model

        # Horizon length
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon

        # Initialize the initial condition constraint (dimension 13)
        ocp.constraints.x0 = np.zeros(13)

        # Cost weighting matrices
        # For the state cost, we use:
        # - position weights (first 3),
        # - velocity weights (next 3),
        # - attitude (quaternion) weights (next 4),
        # - angular velocity weights (last 3).
        Q = np.diag([
            params['tracking_weight_pos'],    # px
            params['tracking_weight_pos'],    # py
            params['tracking_weight_pos'],    # pz
            params['tracking_weight_vel'],    # vx
            params['tracking_weight_vel'],    # vy
            params['tracking_weight_vel'],    # vz
            params['tracking_weight_att'],    # q0
            params['tracking_weight_att'],    # q1
            params['tracking_weight_att'],    # q2
            params['tracking_weight_att'],    # q3
            params['tracking_weight_ang_vel'],# ωx
            params['tracking_weight_ang_vel'],# ωy
            params['tracking_weight_ang_vel'] # ωz
        ])

        # Control cost matrix (same as before)
        R = np.diag([
            params['control_weight'],      # Fx
            params['control_weight'],      # Fy
            params['control_weight'],      # Fz
            params['control_weight']*10,   # τx
            params['control_weight']*10,   # τy
            params['control_weight']*10    # τz
        ])

        # Combined weighting matrix for [x;u]. The overall residual dimension is 13+6 = 19.
        W = np.block([
            [Q, np.zeros((13, 6))],
            [np.zeros((6, 13)), R]
        ])
        ocp.cost.W = W

        # Vx and Vu map state and control to the residual.
        # For a residual of dimension 19:
        ocp.cost.Vx = np.eye(19)[:, :13]   # maps state (13 elements)
        ocp.cost.Vu = np.eye(19)[:, 13:]   # maps control (6 elements)

        # Reference (yref) for the cost, dimension 19 (state + control)
        ocp.cost.yref = np.zeros(19)

        # Control constraints (as before)
        max_force = params['max_force']
        max_torque = params['max_torque']
        ocp.constraints.lbu = np.array([-max_force, -max_force, 0,
                                        -max_torque, -max_torque, -max_torque])
        ocp.constraints.ubu = np.array([ max_force,  max_force,  12*max_force,
                                         max_torque,  max_torque,  max_torque])
        ocp.constraints.idxbu = np.arange(6)

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # Create the solver
        ocp_solver = AcadosOcpSolver(ocp)
        return ocp_solver

    def compute_control(self, state, target_state, initial_guess, dt):
        """
        Compute the control input using the ACADOS OCP solver.

        Parameters:
        -----------
        state: current 13D state
        target_state: desired 13D state (the attitude part should be given as a quaternion)
        initial_guess: a 6D guess for the controls across the horizon
        dt: (not used in this snippet, but could be used to update dt if desired)
        """
        if not self.initialized:
            # If solver isn't built, just return the initial guess
            return initial_guess
        if len(state) == 13:
            # Enforce the initial condition at stage 0
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)
        self.ocp_solver.set(0, "x", state)

        # Build the reference vector for each stage.
        # Typically, we track target_state, while a nominal hover thrust is applied.
        mass = self.params["mass"]
        g    = self.params["gravity"]
        nominal_hover = np.array([0.0, 0.0, mass * g, 0.0, 0.0, 0.0])  # [Fx,Fy,Fz,τx,τy,τz]
        # yref has dimension 19: [target_state (13), nominal_hover (6)]
        yref = np.concatenate([target_state, nominal_hover])

        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            # Optionally, set an initial guess for the control input at each node.
            self.ocp_solver.set(i, "u", initial_guess)

        # Solve the OCP
        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[ACADOS] solver returned status {status}, something went wrong.")
            # Additional error handling can be added here.

        # Extract and return the first control input.
        u_opt = np.array(self.ocp_solver.get(0, "u"))
        return u_opt


def simulate_hexarotor_dynamics(params, mpc, 
                                initial_state=None, 
                                target_state=None, 
                                steps=500):
    """
    Simulate the hexarotor for `steps` steps using the MPC in the loop.

    Parameters:
    -----------
    params: dictionary of parameters, must include 'dt'
    mpc: instance of OneStepMPC
    initial_state: 13D numpy array. If None, it is set to zeros except for the quaternion.
    target_state: 13D numpy array (desired). If None, it is set to zero except for a level attitude.
    steps: number of simulation steps
    """
    # Set default initial state if not provided.
    if initial_state is None:
        # Default: at the origin, zero velocity, level (quaternion [1,0,0,0]), zero angular velocity.
        initial_state = np.zeros(13)
        initial_state[6] = 1.0  # q0 = 1

    if target_state is None:
        target_state = np.zeros(13)
        target_state[6] = 1.0  # level attitude

    # Create a CasADi function for the continuous dynamics f(x,u)
    x_sym = mpc.model.x
    u_sym = mpc.model.u
    f_expl = mpc.model.f_expl_expr
    f_func = cs.Function('f_func', [x_sym, u_sym], [f_expl])

    # Use a small integration time step
    dt = params['dt'] / 100
    state = initial_state.copy()
    state_history = [state]
    control_history = []

    # A reasonable initial guess for the control: hover thrust
    mass = params["mass"]
    g    = params["gravity"]
    init_guess_u = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])

    for _ in range(steps):
        # Get control from the MPC controller
        u_opt = mpc.compute_control(state, target_state, init_guess_u, params['dt'])
        control_history.append(u_opt)

        # Simple Euler integration of the dynamics
        x_dot = np.array(f_func(state, u_opt)).flatten()
        state = state + dt * x_dot
        state_history.append(state)

    return np.array(state_history), np.array(control_history)


def main():
    # Example parameters
    params = {
        'inertia': [0.115125971, 0.116524229, 0.230387752],
        'mass': 7.00,
        'gravity': 9.81,
        'max_force': 20.0,
        'max_torque': 0.05,
        'control_weight': 0.005,
        'tracking_weight_pos': 10,
        'tracking_weight_vel': 3,
        'tracking_weight_att': 80,
        'tracking_weight_ang_vel': 50,
        'smoothness_weight': 0.01,
        'dt': 0.3
    }

    # Build the MPC (quaternion-based)
    mpc = OneStepMPC(params)

    # Define initial and target states (13D).
    # For the quaternion parts, a level attitude is [1, 0, 0, 0].
    initial_state = np.zeros(13)
    initial_state[6] = 1.0  # quaternion: level
    target_state  = np.array([
        1.0, -1.0, 2.0,  # desired position: px, py, pz
        0.0, 0.0, 0.0,   # desired velocity: vx, vy, vz
        1.0, 0.0, 0.0, 0.0,  # desired quaternion: level attitude [1,0,0,0]
        0.0, 0.0, 0.0    # desired angular velocity: ωx, ωy, ωz
    ])

    # Run closed-loop simulation for a number of steps.
    steps = 5000

    state_traj, control_traj = simulate_hexarotor_dynamics(
        params, mpc,
        initial_state=initial_state,
        target_state=target_state,
        steps=steps
    )

    # Create time arrays for plotting. Note: state_traj has length (steps+1), control_traj has length (steps)
    t_states = np.linspace(0, steps*params['dt'], steps+1)
    t_ctrl   = np.linspace(0, (steps-1)*params['dt'], steps)

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    # 1) Plot position (px, py, pz)
    plt.figure(figsize=(8,4))
    plt.plot(t_states, state_traj[:, 0], label='x')
    plt.plot(t_states, state_traj[:, 1], label='y')
    plt.plot(t_states, state_traj[:, 2], label='z')
    plt.title("Position (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    # 2) Plot quaternion components
    plt.figure(figsize=(8,4))
    plt.plot(t_states, state_traj[:, 6], label='q0')
    plt.plot(t_states, state_traj[:, 7], label='q1')
    plt.plot(t_states, state_traj[:, 8], label='q2')
    plt.plot(t_states, state_traj[:, 9], label='q3')
    plt.title("Quaternion Components")
    plt.xlabel("Time (s)")
    plt.ylabel("Quaternion")
    plt.legend()
    plt.grid(True)

    # Alternatively, one could convert the quaternions to Euler angles for plotting.

    # 3) Plot forces (Fx, Fy, Fz)
    plt.figure(figsize=(8,4))
    plt.plot(t_ctrl, control_traj[:, 0], label='Fx')
    plt.plot(t_ctrl, control_traj[:, 1], label='Fy')
    plt.plot(t_ctrl, control_traj[:, 2], label='Fz')
    plt.title("Forces (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)

    # 4) Plot torques (τx, τy, τz)
    plt.figure(figsize=(8,4))
    plt.plot(t_ctrl, control_traj[:, 3], label='τx')
    plt.plot(t_ctrl, control_traj[:, 4], label='τy')
    plt.plot(t_ctrl, control_traj[:, 5], label='τz')
    plt.title("Torques (Nm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Final state:", state_traj[-1])
    print("Final control:", control_traj[-1])


if __name__ == "__main__":
    main()
