import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# ============================================================================
# Multi-Channel Disturbance Observer for Hover Thrust and 3D Torques
# ============================================================================
class MultiDisturbanceObserver:
    def __init__(self, cutoff_freq, gains, dt, acc_min, acc_max):
        """
        A disturbance observer that estimates disturbances on:
          - vertical acceleration (for hover thrust)
          - angular accelerations (for roll, pitch, and yaw torques)
        
        The observer uses a first-order low-pass filter on the disturbance signal
        computed as:
            disturbance = (actual acceleration - desired acceleration)
        
        It then converts the filtered disturbance into a force/torque compensation
        using the provided gains.
        
        Parameters:
        -----------
        cutoff_freq: float
            The cutoff frequency for the low-pass filter (Hz).
        gains: array-like, shape (4,)
            Conversion gains from acceleration to force/torque.
            For example, [mass, I_xx, I_yy, I_zz] or tuned gains.
        dt: float
            The time step.
        acc_min: array-like, shape (4,)
            Minimum allowable disturbance in the acceleration domain (per channel).
        acc_max: array-like, shape (4,)
            Maximum allowable disturbance in the acceleration domain (per channel).
        """
        self.cutoff_freq = cutoff_freq
        self.gains = np.array(gains)   # [gain_Fz, gain_tau_x, gain_tau_y, gain_tau_z]
        self.dt = dt
        self.acc_min = np.array(acc_min)
        self.acc_max = np.array(acc_max)
        
        # Store previous measurement for each channel:
        # Order: [v_z, p, q, r]
        self.prev_meas = np.zeros(4)
        
        # Filtered disturbance (in acceleration domain)
        self.dist_filt = np.zeros(4)
        
        # Compute filter coefficient for a first-order low-pass filter
        self.alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_freq * self.dt)

    def update(self, current_meas, desired_acc):
        """
        Update the disturbance observer.
        
        Parameters:
        -----------
        current_meas: array-like, shape (4,)
            Current measured values [v_z, p, q, r].
        desired_acc: array-like, shape (4,)
            Desired accelerations [a_vz, a_p, a_q, a_r]. For hover/steady rotation, these are typically zeros.
        
        Returns:
        --------
        compensation: ndarray, shape (4,)
            The compensation terms for [Fz, tau_x, tau_y, tau_z].
        """
        current_meas = np.array(current_meas)
        desired_acc = np.array(desired_acc)
        
        # Compute actual acceleration (or angular acceleration) using finite differences.
        actual_acc = (current_meas - self.prev_meas) / self.dt
        
        # Compute disturbance in acceleration domain.
        dist = actual_acc - desired_acc
        
        # Apply first-order low-pass filtering.
        self.dist_filt += self.alpha * (dist - self.dist_filt)
        self.dist_filt = np.clip(self.dist_filt, self.acc_min, self.acc_max)
        
        # Convert the disturbance (acceleration) to force/torque compensation.
        compensation = - self.dist_filt * self.gains
        
        # Update stored measurement.
        self.prev_meas = current_meas.copy()
        
        return compensation

# ============================================================================
# One-Step MPC Class (unchanged from your original design)
# ============================================================================
class OneStepMPC:
    def __init__(self, params):
        """
        Initialize the one-step MPC controller.
        """
        self.params = params
        self.dt = params["dt"]
        self.previous_control = np.zeros(6)  # [Fx, Fy, Fz, tau_x, tau_y, tau_z]
        self.model = self.build_full_hexarotor_model()
        self.horizon = 3  # number of shooting intervals
        self.ocp_solver = self.build_acados_ocp_solver()
        self.initialized = True

    def build_full_hexarotor_model(self):
        """
        Build a 6-DoF hexarotor model in CasADi.
        The state vector is:
          x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        and the control vector is:
          u = [Fx, Fy, Fz, tau_x, tau_y, tau_z]
        """
        # Extract parameters
        I_xx, I_yy, I_zz = self.params["inertia"]
        mass = self.params["mass"]
        g = self.params["gravity"]

        # Define CasADi symbolic variables
        x = cs.MX.sym("x", 12)       # state
        u = cs.MX.sym("u", 6)        # control input
        xdot = cs.MX.sym("xdot", 12)  # state derivative

        # Extract state components
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p_, q_, r_ = x[9], x[10], x[11]

        # Extract control components
        Fx, Fy, Fz = u[0], u[1], u[2]
        tau_x, tau_y, tau_z = u[3], u[4], u[5]

        # Position dynamics
        px_dot = vx
        py_dot = vy
        pz_dot = vz

        # Velocity dynamics (with gravity projection)
        vx_dot = (Fx / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
        vy_dot = (Fy / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
        vz_dot = (Fz / mass) - g * (cs.cos(phi)*cs.cos(theta))

        # Euler angle kinematics
        phi_dot = p_ + q_ * cs.sin(phi)*cs.tan(theta) + r_ * cs.cos(phi)*cs.tan(theta)
        theta_dot = q_ * cs.cos(phi) - r_ * cs.sin(phi)
        psi_dot = q_ * cs.sin(phi)/cs.cos(theta) + r_ * cs.cos(phi)/cs.cos(theta)

        # Angular rate dynamics
        p_dot = (1.0 / I_xx) * (tau_x + (I_yy - I_zz)*q_*r_)
        q_dot = (1.0 / I_yy) * (tau_y + (I_zz - I_xx)*p_*r_)
        r_dot = (1.0 / I_zz) * (tau_z + (I_xx - I_yy)*p_*q_)

        # Combine dynamics
        f_expl = cs.vertcat(px_dot, py_dot, pz_dot,
                            vx_dot, vy_dot, vz_dot,
                            phi_dot, theta_dot, psi_dot,
                            p_dot, q_dot, r_dot)
        f_impl = xdot - f_expl

        # Build the acados model
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
        Build the acados OCP solver for the hexarotor.
        The cost tracks the state error and control usage.
        """
        params = self.params
        ocp = AcadosOcp()
        ocp.model = self.model

        # Set horizon and time duration
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon

        # Initial condition constraint (state dimension is 12)
        ocp.constraints.x0 = np.zeros(12)

        # Cost weighting matrices
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

        R = np.diag([
            params['control_weight'],      # Fx
            params['control_weight'],      # Fy
            params['control_weight'],      # Fz
            params['control_weight']*10,   # tau_x
            params['control_weight']*10,   # tau_y
            params['control_weight']*10    # tau_z
        ])

        # Combined cost weight matrix for residual [x; u]
        W = np.block([
            [Q, np.zeros((12, 6))],
            [np.zeros((6, 12)), R]
        ])
        ocp.cost.W = W

        # Define Vx and Vu so that the cost residual is:
        # residual = Vx*x + Vu*u - yref, dimension 18.
        ocp.cost.Vx = np.eye(18)[:, :12]
        ocp.cost.Vu = np.eye(18)[:, 12:]
        ocp.cost.yref = np.zeros(18)

        # Control constraints
        max_force = params['max_force']
        max_torque = params['max_torque']
        ocp.constraints.lbu = np.array([-max_force, -max_force, 0,
                                        -max_torque, -max_torque, -max_torque])
        ocp.constraints.ubu = np.array([ max_force,  max_force,  12*max_force,
                                         max_torque, max_torque, max_torque])
        ocp.constraints.idxbu = np.arange(6)

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        ocp_solver = AcadosOcpSolver(ocp)
        return ocp_solver

    def compute_control(self, state, target_state, initial_guess, dt):
        """
        Compute the control input using the acados OCP solver.
        
        Parameters:
        -----------
        state: current 12-D state.
        target_state: desired 12-D state.
        initial_guess: initial guess for control over the horizon.
        dt: time step (not directly used here).
        """
        if not self.initialized:
            return initial_guess

        # Enforce the initial condition at stage 0.
        if len(state) == 12:
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)
        self.ocp_solver.set(0, "x", state)

        # Use a nominal hover control for reference.
        mass = self.params["mass"]
        g = self.params["gravity"]
        nominal_hover = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])
        yref = np.concatenate([target_state, nominal_hover])
        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            self.ocp_solver.set(i, "u", initial_guess)

        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[ACADOS] solver returned status {status}, something went wrong.")

        u_opt = np.array(self.ocp_solver.get(0, "u"))
        return u_opt

# ============================================================================
# Simulation Function
# ============================================================================
def simulate_hexarotor_dynamics(sim_params, mpc, dist_observer,
                                initial_state=np.zeros(12), target_state=None, steps=500):
    if target_state is None:
        target_state = np.zeros(12)
    
    # Build the simulation model with mismatched parameters
    sim_model = build_sim_model(sim_params)
    f_func = cs.Function('f_func_sim', [sim_model.x, sim_model.u], [sim_model.f_expl_expr])
    
    sim_dt = sim_params['dt'] / 100  # you might want to use sim_params' dt here
    state = initial_state.copy()
    state_history = [state]
    control_history = []

    mass = sim_params["mass"]
    g = sim_params["gravity"]
    init_guess_u = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])
    
    for _ in range(steps):
        # Compute the control using the MPC (which is built with the original params)
        u_nom = mpc.compute_control(state, target_state, init_guess_u, sim_params['dt'])
        
        # Optionally update the disturbance observer here
        current_meas = [state[5], state[9], state[10], state[11]]
        desired_acc = [0.0, 0.0, 0.0, 0.0]
        compensation = dist_observer.update(current_meas, desired_acc)
        # Apply compensation if desired (currently commented out in your test)
        u_nom[2] += compensation[0]
        u_nom[3] += compensation[1]
        u_nom[4] += compensation[2]
        u_nom[5] += compensation[3]
        
        control_history.append(u_nom)
        x_dot = np.array(f_func(state, u_nom)).flatten()
        state = state + sim_dt * x_dot
        state_history.append(state)
    
    return np.array(state_history), np.array(control_history)

# ============================================================================
# Main Function
# ============================================================================

def build_sim_model(sim_params):
    """
    Build a 6-DoF hexarotor simulation model using sim_params.
    This model is similar to the MPC model but uses the mismatched parameters.
    """
    I_xx, I_yy, I_zz = sim_params["inertia"]
    mass = sim_params["mass"]
    g = sim_params["gravity"]

    x = cs.MX.sym("x", 12)       # state: [px,py,pz, vx,vy,vz, phi,theta,psi, p,q,r]
    u = cs.MX.sym("u", 6)        # control: [Fx,Fy,Fz, tau_x,tau_y,tau_z]
    xdot = cs.MX.sym("xdot", 12)  # state derivative

    # State extraction
    px, py, pz = x[0], x[1], x[2]
    vx, vy, vz = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p_, q_, r_ = x[9], x[10], x[11]

    # Control extraction
    Fx, Fy, Fz = u[0], u[1], u[2]
    tau_x, tau_y, tau_z = u[3], u[4], u[5]

    # Dynamics equations (same structure, but with sim_params)
    px_dot = vx
    py_dot = vy
    pz_dot = vz

    vx_dot = (Fx / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
    vy_dot = (Fy / mass) - g * (cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
    vz_dot = (Fz / mass) - g * (cs.cos(phi)*cs.cos(theta))

    phi_dot   = p_ + q_ * cs.sin(phi)*cs.tan(theta) + r_ * cs.cos(phi)*cs.tan(theta)
    theta_dot = q_ * cs.cos(phi) - r_ * cs.sin(phi)
    psi_dot   = q_ * cs.sin(phi)/cs.cos(theta) + r_ * cs.cos(phi)/cs.cos(theta)

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
    model.name = "sim_hexarotor_model"
    return model
def main():
    # Define example parameters.
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
    sim_params["mass"] = 18.5  # e.g., simulate a heavier system
    sim_params["inertia"] = [0.06000, 0.000525, 0.00150]  # different inertias
    # Create the MPC instance.
    mpc = OneStepMPC(params)

    # Instantiate the multi-channel disturbance observer.
    # For gains, a typical choice is to use the mass for Fz and moments of inertia for torques.
    gains = [params["mass"], 0.115125971, 0.116524229, 0.230387752]
    # Set reasonable acceleration limits per channel (tune these as needed).
    acc_min = [-1.0, -0.5, -0.5, -0.5]
    acc_max = [ 1.0,  0.5,  0.5,  0.5]
    dist_observer = MultiDisturbanceObserver(
        cutoff_freq=5.0,
        gains=gains,
        dt=params["dt"],
        acc_min=acc_min,
        acc_max=acc_max
    )

    # Define initial and target states (12-D).
    initial_state = np.zeros(12)
    target_state = np.array([
        1.0, -1.0, 2.0,  # Desired position: px, py, pz
        0.0, 0.0, 0.0,   # Desired velocity: vx, vy, vz
        0.0, 0.0, 0.0,   # Desired orientation: phi, theta, psi
        0.0, 0.0, 0.0    # Desired angular rates: p, q, r
    ])

    steps = 5000
    state_traj, control_traj = simulate_hexarotor_dynamics(
        sim_params, mpc, dist_observer,
        initial_state=initial_state,
        target_state=target_state,
        steps=steps
    )

    # Create time arrays for plotting.
    t_states = np.linspace(0, steps * params['dt'], steps + 1)
    t_ctrl   = np.linspace(0, (steps - 1) * params['dt'], steps)

    # --- Plotting ---
    # 1) Positions.
    plt.figure(figsize=(8, 4))
    plt.plot(t_states, state_traj[:, 0], label='x')
    plt.plot(t_states, state_traj[:, 1], label='y')
    plt.plot(t_states, state_traj[:, 2], label='z')
    plt.title("Position (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    # 2) Orientations (in degrees).
    roll  = state_traj[:, 6] * 180.0 / np.pi
    pitch = state_traj[:, 7] * 180.0 / np.pi
    yaw   = state_traj[:, 8] * 180.0 / np.pi
    plt.figure(figsize=(8, 4))
    plt.plot(t_states, roll, label='roll (deg)')
    plt.plot(t_states, pitch, label='pitch (deg)')
    plt.plot(t_states, yaw, label='yaw (deg)')
    plt.title("Orientation Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True)

    # 3) Control Forces.
    plt.figure(figsize=(8, 4))
    plt.plot(t_ctrl, control_traj[:, 0], label='Fx')
    plt.plot(t_ctrl, control_traj[:, 1], label='Fy')
    plt.plot(t_ctrl, control_traj[:, 2], label='Fz')
    plt.title("Forces (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)

    # 4) Control Torques.
    plt.figure(figsize=(8, 4))
    plt.plot(t_ctrl, control_traj[:, 3], label='tau_x')
    plt.plot(t_ctrl, control_traj[:, 4], label='tau_y')
    plt.plot(t_ctrl, control_traj[:, 5], label='tau_z')
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
