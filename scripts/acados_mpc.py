import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


class OneStepMPC:
    def __init__(self, params):
        """
        Initialize the one-step MPC controller.

        Parameters:
        -----------
        params: Dictionary containing system parameters such as inertia, mass, gravity, etc.
        """
        self.params = params
        self.dt = params["dt"]
        self.previous_control = np.zeros(6)  # Store the previous control input (if you want smoothness, etc.)

        # Build the full hexarotor model
        self.model = self.build_full_hexarotor_model()

        # Set up horizon
        self.horizon = 2  # e.g. 29 steps -> (29+1) knot points if you use discrete shooting
        self.ocp_solver = self.build_acados_ocp_solver()
        self.initialized = True

    def build_full_hexarotor_model(self):
        """
        Build a 6-DoF hexarotor model in CasADi that is consistent with
        the 'RK4-based' code snippet (full Euler angles, gravity projection, etc.).

        State vector x = [ px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r ]
            px, py, pz : position in world frame
            vx, vy, vz : velocity in world frame
            phi, theta, psi : roll, pitch, yaw (Euler angles)
            p, q, r : angular rates in body frame

        Control vector u = [ Fx, Fy, Fz, tau_x, tau_y, tau_z ]
            Fx, Fy, Fz : forces in body frame
            tau_x, tau_y, tau_z : torques about body axes
        """
        # Extract parameters
        I_xx, I_yy, I_zz = self.params["inertia"]   # moments of inertia
        mass = self.params["mass"]
        g    = self.params["gravity"]

        # Define CasADi variables
        x = cs.MX.sym("x", 12)     # state
        u = cs.MX.sym("u", 6)      # control
        xdot = cs.MX.sym("xdot", 12)  # for implicit form

        # Extract states for readability
        px    = x[0]
        py    = x[1]
        pz    = x[2]
        vx    = x[3]
        vy    = x[4]
        vz    = x[5]
        phi   = x[6]   # roll
        theta = x[7]   # pitch
        psi   = x[8]   # yaw
        p_    = x[9]   # roll rate
        q_    = x[10]  # pitch rate
        r_    = x[11]  # yaw rate

        # Extract controls for readability
        Fx    = u[0]
        Fy    = u[1]
        Fz    = u[2]
        tau_x = u[3]
        tau_y = u[4]
        tau_z = u[5]

        # -- Position derivatives
        px_dot = vx
        py_dot = vy
        pz_dot = vz

        # -- Velocity derivatives (with gravity projection via Euler angles)
        vx_dot = (Fx / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
        vy_dot = (Fy / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
        vz_dot = (Fz / mass) - g * (cs.cos(phi)*cs.cos(theta))

        # -- Euler angle kinematics
        phi_dot = p_ + q_ * cs.sin(phi)*cs.tan(theta) + r_ * cs.cos(phi)*cs.tan(theta)
        theta_dot = q_ * cs.cos(phi) - r_ * cs.sin(phi)
        psi_dot = q_ * cs.sin(phi)/cs.cos(theta) + r_ * cs.cos(phi)/cs.cos(theta)

        # -- Angular rates
        p_dot = (1.0 / I_xx) * (tau_x + (I_yy - I_zz)*q_*r_)
        q_dot = (1.0 / I_yy) * (tau_y + (I_zz - I_xx)*p_*r_)
        r_dot = (1.0 / I_zz) * (tau_z + (I_xx - I_yy)*p_*q_)

        # Collect the explicit dynamics f(x,u)
        f_expl = cs.vertcat(
            px_dot,
            py_dot,
            pz_dot,
            vx_dot,
            vy_dot,
            vz_dot,
            phi_dot,
            theta_dot,
            psi_dot,
            p_dot,
            q_dot,
            r_dot
        )

        # Implicit form: xdot - f_expl = 0
        f_impl = xdot - f_expl

        # Build AcadosModel
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "full_hexarotor_model"

        return model

    def build_acados_ocp_solver(self):
        """
        Build the ACADOS OCP solver for the hexarotor.

        The cost tracks both state error and control usage, with constraints on controls.
        """
        params = self.params
        ocp = AcadosOcp()
        ocp.model = self.model

        # Horizon length
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon

        # Initialize the initial condition constraint
        ocp.constraints.x0 = np.zeros(12)

        # Cost weighting matrices
        Q = np.diag([
            params['tracking_weight_pos']*8,    # px
            params['tracking_weight_pos']*8,    # py
            params['tracking_weight_pos']*26,    # pz
            params['tracking_weight_vel'],    # vx
            params['tracking_weight_vel'],    # vy
            params['tracking_weight_vel'],    # vz
            params['tracking_weight_att']*16,    # phi
            params['tracking_weight_att']*16,    # theta
            params['tracking_weight_att']*16,    # psi
            params['tracking_weight_ang_vel'],# p
            params['tracking_weight_ang_vel'],# q
            params['tracking_weight_ang_vel'] # r
        ])

        R = np.diag([
            params['control_weight'],  # Fx
            params['control_weight'],  # Fy
            params['control_weight'],  # Fz
            params['control_weight']*10,  # tau_x
            params['control_weight']*10,  # tau_y
            params['control_weight']*10   # tau_z
        ])

        # Combined W for [x;u]
        W = np.block([
            [Q, np.zeros((12, 6))],
            [np.zeros((6, 12)), R]
        ])
        ocp.cost.W = W

        # Vx, Vu define how we pick (x,u) into the cost function
        # The dimension of the cost residual is 12+6 = 18
        # So we want Vx to shape that 18D residual from x, and Vu from u
        ocp.cost.Vx = np.eye(18)[:, :12]   # first 12 columns map to state
        ocp.cost.Vu = np.eye(18)[:, 12:]   # last 6 columns map to control

        # Reference (yref), dimension 18
        ocp.cost.yref = np.zeros(18)

        # Control constraints
        max_force = params['max_force']
        max_torque = params['max_torque']
        # For example, let Fz > 0 if you want to ensure "upward" thrust,
        # but we can set to 0 or negative if you want the possibility to pull down.
        # Adjust as needed:
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
        Compute the control input using ACADOS OCP solver.

        - state: current 12D state
        - target_state: desired 12D reference
        - initial_guess: a 6D guess for the controls (Fx,Fy,Fz,Tx,Ty,Tz) across the horizon
        """
        if not self.initialized:
            # If solver isn't built, just return the initial guess
            return initial_guess
        if len(state) == 12:
            # Force the solver to treat x0 == state at stage 0
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)
        # Set the initial condition
        self.ocp_solver.set(0, "x", state)

        # Build a reference for each stage in the horizon
        # Typically we want to track target_state, while also wanting hover thrust, etc.
        mass = self.params["mass"]
        g    = self.params["gravity"]
        nominal_hover = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])  # typical hover guess
        yref = np.concatenate([target_state, nominal_hover])

        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            # Optionally set an initial guess for the controls at each node
            self.ocp_solver.set(i, "u", initial_guess)

        # Solve the OCP
        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[ACADOS] solver returned status {status}, something went wrong.")
            # You might want to handle infeasibility or solver issues here.

        # Extract the first control
        u_opt = np.array(self.ocp_solver.get(0, "u"))
        return u_opt


def simulate_hexarotor_dynamics(params, mpc, 
                                initial_state=np.zeros(12), 
                                target_state=None, 
                                steps=500):
    """
    Simulate the hexarotor for `steps` steps using the MPC in the loop.
    
    - params: dictionary of parameters, includes 'dt'
    - mpc: instance of OneStepMPC
    - initial_state: 12D numpy array
    - target_state: 12D numpy array (desired)
    - steps: number of simulation steps
    """
    if target_state is None:
        target_state = np.zeros(12)

    # Build a CasADi function for the continuous dynamics f(x,u).
    x_sym = mpc.model.x
    u_sym = mpc.model.u
    f_expl = mpc.model.f_expl_expr
    f_func = cs.Function('f_func', [x_sym, u_sym], [f_expl])

    dt = params['dt'] / 100
    state = initial_state.copy()
    state_history = [state]
    control_history = []

    # A reasonable initial guess for the control: "hover" + zero for lateral
    mass = params["mass"]
    g    = params["gravity"]
    init_guess_u = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])

    for _ in range(steps):
        # Get control from MPC
        u_opt = mpc.compute_control(state, target_state, init_guess_u, 0.02)
        control_history.append(u_opt)

        # Euler integration
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

    # Build the MPC
    mpc = OneStepMPC(params)

    # Initial and target states
    initial_state = np.zeros(12)
    target_state  = np.array([
        1.0, -1.0, 2.0,  # px, py, pz
        0.0, 0.0, 0.0,  # vx, vy, vz
        0.0, 0.0, 0.0,  # phi, theta, psi
        0.0, 0.0, 0.0   # p, q, r
    ])

    # Run closed-loop simulation for 500 steps
    steps = 5000

    state_traj, control_traj = simulate_hexarotor_dynamics(
        params, mpc,
        initial_state=initial_state,
        target_state=target_state,
        steps=steps
    )

    # Create time arrays for plotting
    # state_traj has length (steps+1), control_traj has length (steps)
    t_states = np.linspace(0, steps*params['dt'], steps+1)
    t_ctrl   = np.linspace(0, (steps-1)*params['dt'], steps)

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    # 1) plot x, y, z
    plt.figure(figsize=(8,4))
    plt.plot(t_states, state_traj[:, 0], label='x')
    plt.plot(t_states, state_traj[:, 1], label='y')
    plt.plot(t_states, state_traj[:, 2], label='z')
    plt.title("Position (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    # 2) plot roll, pitch, yaw
    # convert from rad to deg if you like:
    roll  = state_traj[:, 6]  * 180.0/np.pi
    pitch = state_traj[:, 7]  * 180.0/np.pi
    yaw   = state_traj[:, 8]  * 180.0/np.pi

    plt.figure(figsize=(8,4))
    plt.plot(t_states, roll, label='roll (deg)')
    plt.plot(t_states, pitch, label='pitch (deg)')
    plt.plot(t_states, yaw, label='yaw (deg)')
    plt.title("Orientation Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True)

    # 3) plot Fx, Fy, Fz
    plt.figure(figsize=(8,4))
    plt.plot(t_ctrl, control_traj[:, 0], label='Fx')
    plt.plot(t_ctrl, control_traj[:, 1], label='Fy')
    plt.plot(t_ctrl, control_traj[:, 2], label='Fz')
    plt.title("Forces (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)

    # 4) plot Mx, My, Mz (i.e., tau_x, tau_y, tau_z)
    plt.figure(figsize=(8,4))
    plt.plot(t_ctrl, control_traj[:, 3], label='Mx / tau_x')
    plt.plot(t_ctrl, control_traj[:, 4], label='My / tau_y')
    plt.plot(t_ctrl, control_traj[:, 5], label='Mz / tau_z')
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
