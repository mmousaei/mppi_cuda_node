import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# =============================================================================
# 1. NOMINAL MPC (12-D state)
# =============================================================================
class OneStepMPC_Nominal:
    def __init__(self, params):
        """
        MPC controller using the nominal 12-D hexarotor model.
        """
        self.params = params
        self.dt = params["dt"]
        self.previous_control = np.zeros(6)  # [Fx, Fy, Fz, tau_x, tau_y, tau_z]
        self.model = self.build_hexarotor_model()  # 12-D model
        self.horizon = 3
        self.ocp_solver = self.build_acados_ocp_solver()
        self.initialized = True

    def build_hexarotor_model(self):
        # Extract parameters (nominal)
        I_xx, I_yy, I_zz = self.params["inertia"]
        mass = self.params["mass"]
        g = self.params["gravity"]

        # Define CasADi variables for the 12-D state model
        x = cs.MX.sym("x", 12)
        u = cs.MX.sym("u", 6)
        xdot = cs.MX.sym("xdot", 12)

        # States: [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p_, q_, r_ = x[9], x[10], x[11]

        # Controls: [Fx, Fy, Fz, tau_x, tau_y, tau_z]
        Fx, Fy, Fz = u[0], u[1], u[2]
        tau_x, tau_y, tau_z = u[3], u[4], u[5]

        # Dynamics:
        px_dot = vx
        py_dot = vy
        pz_dot = vz
        vx_dot = (Fx / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
        vy_dot = (Fy / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
        vz_dot = (Fz / mass) - g * (cs.cos(phi)*cs.cos(theta))
        phi_dot = p_ + q_ * cs.sin(phi)*cs.tan(theta) + r_ * cs.cos(phi)*cs.tan(theta)
        theta_dot = q_ * cs.cos(phi) - r_ * cs.sin(phi)
        psi_dot = q_ * cs.sin(phi)/cs.cos(theta) + r_ * cs.cos(phi)/cs.cos(theta)
        p_dot = (1.0 / I_xx) * (tau_x + (I_yy - I_zz)*q_*r_)
        q_dot = (1.0 / I_yy) * (tau_y + (I_zz - I_xx)*p_*r_)
        r_dot = (1.0 / I_zz) * (tau_z + (I_xx - I_yy)*p_*q_)

        f_expl = cs.vertcat(px_dot, py_dot, pz_dot,
                            vx_dot, vy_dot, vz_dot,
                            phi_dot, theta_dot, psi_dot,
                            p_dot, q_dot, r_dot)
        f_impl = xdot - f_expl

        model = AcadosModel()
        model.x = x
        model.u = u
        model.xdot = xdot
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.name = "hexarotor_model_nominal"
        return model

    def build_acados_ocp_solver(self):
        params = self.params
        ocp = AcadosOcp()
        ocp.model = self.model

        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon
        ocp.constraints.x0 = np.zeros(12)

        Q = np.diag([
            params['tracking_weight_pos'],
            params['tracking_weight_pos'],
            params['tracking_weight_pos']*16,
            params['tracking_weight_vel'],
            params['tracking_weight_vel'],
            params['tracking_weight_vel'],
            params['tracking_weight_att'],
            params['tracking_weight_att'],
            params['tracking_weight_att'],
            params['tracking_weight_ang_vel'],
            params['tracking_weight_ang_vel'],
            params['tracking_weight_ang_vel']
        ])

        R = np.diag([
            params['control_weight'],
            params['control_weight'],
            params['control_weight'],
            params['control_weight']*10,
            params['control_weight']*10,
            params['control_weight']*10
        ])

        W = np.block([
            [Q, np.zeros((12, 6))],
            [np.zeros((6, 12)), R]
        ])
        ocp.cost.W = W
        ocp.cost.Vx = np.eye(18)[:, :12]
        ocp.cost.Vu = np.eye(18)[:, 12:]
        ocp.cost.yref = np.zeros(18)

        max_force = params['max_force']
        max_torque = params['max_torque']
        ocp.constraints.lbu = np.array([-max_force, -max_force, 0,
                                        -max_torque, -max_torque, -max_torque])
        ocp.constraints.ubu = np.array([ max_force,  max_force, 12*max_force,
                                         max_torque, max_torque, max_torque])
        ocp.constraints.idxbu = np.arange(6)

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        ocp_solver = AcadosOcpSolver(ocp)
        return ocp_solver

    def compute_control(self, state, target_state, initial_guess, dt):
        if not self.initialized:
            return initial_guess

        # Enforce the initial condition at stage 0
        self.ocp_solver.set(0, "lbx", state)
        self.ocp_solver.set(0, "ubx", state)
        self.ocp_solver.set(0, "x", state)

        mass = self.params["mass"]
        g = self.params["gravity"]
        nominal_hover = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])
        yref = np.concatenate([target_state, nominal_hover])
        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            self.ocp_solver.set(i, "u", initial_guess)

        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[ACADOS] solver returned status {status}")

        u_opt = np.array(self.ocp_solver.get(0, "u"))
        return u_opt

# =============================================================================
# 2. ADAPTIVE MPC (Augmented, 16-D state)
# =============================================================================
class OneStepMPC_Adaptive:
    def __init__(self, params):
        """
        MPC controller using an augmented hexarotor model.
        The state is augmented as:
            x_full = [ x_phys (12-D); d (4-D) ]
        where d are additive disturbance/adaptation states that affect the effective
        vertical force and torques.
        """
        self.params = params
        self.dt = params["dt"]
        self.previous_control = np.zeros(6)  # control remains 6-D
        self.model = self.build_augmented_hexarotor_model()
        self.horizon = 3
        self.ocp_solver = self.build_acados_ocp_solver()
        self.initialized = True

    def build_augmented_hexarotor_model(self):
        # Extract nominal parameters (used by the controller)
        I_xx, I_yy, I_zz = self.params["inertia"]
        mass = self.params["mass"]
        g = self.params["gravity"]

        # Augmented state: first 12 are physical, last 4 are adaptation offsets.
        x_full = cs.MX.sym("x", 16)
        u = cs.MX.sym("u", 6)
        xdot_full = cs.MX.sym("xdot", 16)

        # Physical state and adaptation offsets.
        x_phys = x_full[0:12]
        d = x_full[12:16]

        # Extract physical states (same as before)
        px, py, pz = x_phys[0], x_phys[1], x_phys[2]
        vx, vy, vz = x_phys[3], x_phys[4], x_phys[5]
        phi, theta, psi = x_phys[6], x_phys[7], x_phys[8]
        p_, q_, r_ = x_phys[9], x_phys[10], x_phys[11]

        # Controls remain the same.
        Fx, Fy, Fz = u[0], u[1], u[2]
        tau_x, tau_y, tau_z = u[3], u[4], u[5]

        # Physical dynamics as before, but use adapted control:
        # Effective vertical force and torques include the d-offsets.
        px_dot = vx
        py_dot = vy
        pz_dot = vz
        vx_dot = (Fx / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
        vy_dot = (Fy / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
        vz_dot = ((Fz + d[0]) / mass) - g * (cs.cos(phi)*cs.cos(theta))
        phi_dot = p_ + q_ * cs.sin(phi)*cs.tan(theta) + r_ * cs.cos(phi)*cs.tan(theta)
        theta_dot = q_ * cs.cos(phi) - r_ * cs.sin(phi)
        psi_dot = q_ * cs.sin(phi)/cs.cos(theta) + r_ * cs.cos(phi)/cs.cos(theta)
        p_dot = (1.0 / I_xx) * (tau_x + d[1] + (I_yy - I_zz)*q_*r_)
        q_dot = (1.0 / I_yy) * (tau_y + d[2] + (I_zz - I_xx)*p_*r_)
        r_dot = (1.0 / I_zz) * (tau_z + d[3] + (I_xx - I_yy)*p_*q_)

        f_expl_phys = cs.vertcat(px_dot, py_dot, pz_dot,
                                 vx_dot, vy_dot, vz_dot,
                                 phi_dot, theta_dot, psi_dot,
                                 p_dot, q_dot, r_dot)
        # Adaptation states are constant: d_dot = 0.
        d_dot = cs.DM.zeros(4)
        f_expl_full = cs.vertcat(f_expl_phys, d_dot)
        f_impl = xdot_full - f_expl_full

        model = AcadosModel()
        model.x = x_full
        model.u = u
        model.xdot = xdot_full
        model.f_expl_expr = f_expl_full
        model.f_impl_expr = f_impl
        model.name = "hexarotor_model_adaptive"
        return model

    def build_acados_ocp_solver(self):
        params = self.params
        ocp = AcadosOcp()
        ocp.model = self.model

        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon
        # Initial condition is 16-D.
        ocp.constraints.x0 = np.zeros(16)

        # Cost weighting:
        Q = np.diag([
            params['tracking_weight_pos'],  # px
            params['tracking_weight_pos'],  # py
            params['tracking_weight_pos']*16,  # pz
            params['tracking_weight_vel'],  # vx
            params['tracking_weight_vel'],  # vy
            params['tracking_weight_vel'],  # vz
            params['tracking_weight_att'],  # phi
            params['tracking_weight_att'],  # theta
            params['tracking_weight_att'],  # psi
            params['tracking_weight_ang_vel'],  # p
            params['tracking_weight_ang_vel'],  # q
            params['tracking_weight_ang_vel']   # r
        ])
        # We penalize the adaptation states lightly:
        Q_ad = np.diag([0.1, 0.1, 0.1, 0.1])
        Q_ext = np.block([
            [Q, np.zeros((12,4))],
            [np.zeros((4,12)), Q_ad]
        ])

        R = np.diag([
            params['control_weight'],
            params['control_weight'],
            params['control_weight'],
            params['control_weight']*10,
            params['control_weight']*10,
            params['control_weight']*10
        ])

        W = np.block([
            [Q_ext, np.zeros((16,6))],
            [np.zeros((6,16)), R]
        ])
        ocp.cost.W = W

        ocp.cost.Vx = np.eye(22)[:, :16]
        ocp.cost.Vu = np.eye(22)[:, 16:]
        ocp.cost.yref = np.zeros(22)

        max_force = params['max_force']
        max_torque = params['max_torque']
        ocp.constraints.lbu = np.array([-max_force, -max_force, 0,
                                        -max_torque, -max_torque, -max_torque])
        ocp.constraints.ubu = np.array([ max_force,  max_force, 12*max_force,
                                         max_torque, max_torque, max_torque])
        ocp.constraints.idxbu = np.arange(6)

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        ocp_solver = AcadosOcpSolver(ocp)
        return ocp_solver

    def compute_control(self, state, target_state, initial_guess, dt):
        if not self.initialized:
            return initial_guess

        # If state is provided as 12-D, augment it with zeros for the adaptation states.
        if len(state) == 12:
            state = np.concatenate([state, np.zeros(4)])
        self.ocp_solver.set(0, "lbx", state)
        self.ocp_solver.set(0, "ubx", state)
        self.ocp_solver.set(0, "x", state)

        mass = self.params["mass"]
        g = self.params["gravity"]
        nominal_hover = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])
        target_state_ext = np.concatenate([target_state, np.zeros(4)])
        yref = np.concatenate([target_state_ext, nominal_hover])
        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            self.ocp_solver.set(i, "u", initial_guess)

        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[ACADOS] solver returned status {status}")

        u_opt = np.array(self.ocp_solver.get(0, "u"))
        return u_opt

# =============================================================================
# Simulation Models
# =============================================================================
def build_sim_model_nominal(sim_params):
    """
    Build a 12-D simulation model (the true plant) using sim_params (which may be mismatched).
    """
    I_xx, I_yy, I_zz = sim_params["inertia"]
    mass = sim_params["mass"]
    g = sim_params["gravity"]

    x = cs.MX.sym("x", 12)
    u = cs.MX.sym("u", 6)
    xdot = cs.MX.sym("xdot", 12)

    px, py, pz = x[0], x[1], x[2]
    vx, vy, vz = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p_, q_, r_ = x[9], x[10], x[11]

    Fx, Fy, Fz = u[0], u[1], u[2]
    tau_x, tau_y, tau_z = u[3], u[4], u[5]

    px_dot = vx
    py_dot = vy
    pz_dot = vz
    vx_dot = (Fx / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
    vy_dot = (Fy / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
    vz_dot = (Fz / mass) - g * (cs.cos(phi)*cs.cos(theta))
    phi_dot = p_ + q_ * cs.sin(phi)*cs.tan(theta) + r_ * cs.cos(phi)*cs.tan(theta)
    theta_dot = q_ * cs.cos(phi) - r_ * cs.sin(phi)
    psi_dot = q_ * cs.sin(phi)/cs.cos(theta) + r_ * cs.cos(phi)/cs.cos(theta)
    p_dot = (1.0 / I_xx) * (tau_x + (I_yy - I_zz)*q_*r_)
    q_dot = (1.0 / I_yy) * (tau_y + (I_zz - I_xx)*p_*r_)
    r_dot = (1.0 / I_zz) * (tau_z + (I_xx - I_yy)*p_*q_)

    f_expl = cs.vertcat(px_dot, py_dot, pz_dot,
                        vx_dot, vy_dot, vz_dot,
                        phi_dot, theta_dot, psi_dot,
                        p_dot, q_dot, r_dot)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.x = x
    model.u = u
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.name = "sim_model_nominal"
    return model

def build_sim_model_adaptive(sim_params):
    """
    Build an augmented simulation model (16-D) that mimics the adaptive MPC model.
    The physical dynamics use sim_params (mismatched) and the adaptation states have zero dynamics.
    """
    I_xx, I_yy, I_zz = sim_params["inertia"]
    mass = sim_params["mass"]
    g = sim_params["gravity"]

    # Augmented state: [x_phys (12); d (4)]
    x_full = cs.MX.sym("x", 16)
    u = cs.MX.sym("u", 6)
    xdot_full = cs.MX.sym("xdot", 16)

    x_phys = x_full[0:12]
    d = x_full[12:16]

    px, py, pz = x_phys[0], x_phys[1], x_phys[2]
    vx, vy, vz = x_phys[3], x_phys[4], x_phys[5]
    phi, theta, psi = x_phys[6], x_phys[7], x_phys[8]
    p_, q_, r_ = x_phys[9], x_phys[10], x_phys[11]

    Fx, Fy, Fz = u[0], u[1], u[2]
    tau_x, tau_y, tau_z = u[3], u[4], u[5]

    px_dot = vx
    py_dot = vy
    pz_dot = vz
    vx_dot = (Fx / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
    vy_dot = (Fy / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
    vz_dot = ((Fz + d[0]) / mass) - g * (cs.cos(phi)*cs.cos(theta))
    phi_dot = p_ + q_ * cs.sin(phi)*cs.tan(theta) + r_ * cs.cos(phi)*cs.tan(theta)
    theta_dot = q_ * cs.cos(phi) - r_ * cs.sin(phi)
    psi_dot = q_ * cs.sin(phi)/cs.cos(theta) + r_ * cs.cos(phi)/cs.cos(theta)
    p_dot = (1.0 / I_xx) * (tau_x + d[1] + (I_yy - I_zz)*q_*r_)
    q_dot = (1.0 / I_yy) * (tau_y + d[2] + (I_zz - I_xx)*p_*r_)
    r_dot = (1.0 / I_zz) * (tau_z + d[3] + (I_xx - I_yy)*p_*q_)

    f_expl_phys = cs.vertcat(px_dot, py_dot, pz_dot,
                             vx_dot, vy_dot, vz_dot,
                             phi_dot, theta_dot, psi_dot,
                             p_dot, q_dot, r_dot)
    d_dot = cs.DM.zeros(4)
    f_expl_full = cs.vertcat(f_expl_phys, d_dot)
    f_impl = xdot_full - f_expl_full

    model = AcadosModel()
    model.x = x_full
    model.u = u
    model.xdot = xdot_full
    model.f_expl_expr = f_expl_full
    model.f_impl_expr = f_impl
    model.name = "sim_model_adaptive"
    return model

# =============================================================================
# Simulation Functions
# =============================================================================
def simulate_nominal_mpc(sim_params, mpc, initial_state, target_state, steps):
    """
    Simulate the closed-loop dynamics using the nominal MPC.
    The simulation plant is built using sim_params (mismatched) and is 12-D.
    """
    sim_model = build_sim_model_nominal(sim_params)
    f_func = cs.Function('f_func_sim', [sim_model.x, sim_model.u], [sim_model.f_expl_expr])
    sim_dt = sim_params['dt'] / 100

    state = initial_state.copy()
    state_history = [state]
    control_history = []

    mass = sim_params["mass"]
    g = sim_params["gravity"]
    init_guess_u = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])

    for _ in range(steps):
        u_nom = mpc.compute_control(state, target_state, init_guess_u, sim_params['dt'])
        control_history.append(u_nom)
        x_dot = np.array(f_func(state, u_nom)).flatten()
        state = state + sim_dt * x_dot
        state_history.append(state)
    return np.array(state_history), np.array(control_history)

def simulate_adaptive_mpc(sim_params, mpc, initial_state, target_state, steps):
    """
    Simulate the closed-loop dynamics using the adaptive MPC.
    The simulation plant is augmented (16-D) to mimic the adaptive controller.
    The initial state for the adaptive simulation is [x_phys; zeros].
    """
    sim_model = build_sim_model_adaptive(sim_params)
    f_func = cs.Function('f_func_sim_adapt', [sim_model.x, sim_model.u], [sim_model.f_expl_expr])
    sim_dt = sim_params['dt'] / 100

    # Augment initial state (12-D physical state plus 4 zeros)
    state = np.concatenate([initial_state, np.zeros(4)])
    state_history = [state]
    control_history = []

    mass = sim_params["mass"]
    g = sim_params["gravity"]
    init_guess_u = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])

    for _ in range(steps):
        u_nom = mpc.compute_control(state, target_state, init_guess_u, sim_params['dt'])
        control_history.append(u_nom)
        x_dot = np.array(f_func(state, u_nom)).flatten()
        state = state + sim_dt * x_dot
        state_history.append(state)
    return np.array(state_history), np.array(control_history)

# =============================================================================
# Main Function: Run both Nominal and Adaptive MPC with intentional mismatch
# =============================================================================
def main():
    # Nominal parameters used by the controller
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

    # Simulation parameters (intentional mismatch)
    sim_params = params.copy()
    sim_params["mass"] = 18.5  # Heavier system
    sim_params["inertia"] = [0.06000, 0.000525, 0.00150]  # Different inertias

    # Initial and target states (physical, 12-D)
    initial_state = np.zeros(12)
    target_state = np.array([
        1.0, -1.0, 2.0,   # Desired position
        0.0, 0.0, 0.0,    # Desired velocity
        0.0, 0.0, 0.0,    # Desired orientation
        0.0, 0.0, 0.0     # Desired angular rates
    ])

    steps = 5000

    # Create controllers
    mpc_nominal = OneStepMPC_Nominal(params)
    mpc_adaptive = OneStepMPC_Adaptive(params)

    # Run simulations
    state_traj_nom, control_traj_nom = simulate_nominal_mpc(sim_params, mpc_nominal,
                                                            initial_state, target_state, steps)
    state_traj_adapt, control_traj_adapt = simulate_adaptive_mpc(sim_params, mpc_adaptive,
                                                                 initial_state, target_state, steps)

    # Create time arrays for plotting
    t_states_nom = np.linspace(0, steps * params['dt'], steps + 1)
    t_states_adapt = np.linspace(0, steps * params['dt'], steps + 1)
    t_ctrl = np.linspace(0, (steps - 1) * params['dt'], steps)

    # ----------------- Plot Positions -----------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_states_nom, state_traj_nom[:, 0], label='x')
    plt.plot(t_states_nom, state_traj_nom[:, 1], label='y')
    plt.plot(t_states_nom, state_traj_nom[:, 2], label='z')
    plt.title("Nominal MPC - Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # For adaptive, plot the physical states (first 3 of augmented state)
    plt.plot(t_states_adapt, state_traj_adapt[:, 0], label='x')
    plt.plot(t_states_adapt, state_traj_adapt[:, 1], label='y')
    plt.plot(t_states_adapt, state_traj_adapt[:, 2], label='z')
    plt.title("Adaptive MPC - Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_states_nom, state_traj_nom[:, 6], label='x')
    plt.plot(t_states_nom, state_traj_nom[:, 7], label='y')
    plt.plot(t_states_nom, state_traj_nom[:, 8], label='z')
    plt.title("Nominal MPC - Attitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Attitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # For adaptive, plot the physical states (first 3 of augmented state)
    plt.plot(t_states_adapt, state_traj_adapt[:, 6], label='x')
    plt.plot(t_states_adapt, state_traj_adapt[:, 7], label='y')
    plt.plot(t_states_adapt, state_traj_adapt[:, 8], label='z')
    plt.title("Adaptive MPC - Attitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Attitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # (Additional plotting for orientation, controls, etc. can be added similarly.)
    print("Final Nominal MPC state (physical):", state_traj_nom[-1])
    print("Final Adaptive MPC physical state:", state_traj_adapt[-1, :12])
    print("Final Nominal MPC control:", control_traj_nom[-1])
    print("Final Adaptive MPC control:", control_traj_adapt[-1])

if __name__ == "__main__":
    main()
