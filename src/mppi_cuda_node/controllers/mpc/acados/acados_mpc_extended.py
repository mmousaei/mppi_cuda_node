import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import os

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

class MPC:
    def __init__(self, params):
        """
        Initialize the offset-free MPC controller with state augmentation.
        The augmented state is X = [ x; z ], where x (12D) is the original state,
        and z (12D) integrates the error (x - r) to drive steady-state offset to zero.
        """
        self.params = params
        self.dt = params["dt"]
        self.previous_control = np.zeros(6)
        # Initialize the integrator state (z), which is 12D.
        self.int_state = np.zeros(12)

        # Build the augmented hexarotor model.
        self.model = self.build_augmented_hexarotor_model()

        # Set the horizon
        self.horizon = params["horizon"]
        self.ocp_solver = self.build_acados_ocp_solver()
        self.initialized = True

    def update_parameters(self, new_params):
        self.params = new_params.copy()
        self.__init__(self.params)

    def build_augmented_hexarotor_model(self):
        """
        Build an augmented 6-DoF hexarotor model in CasADi.
        The original state x is 12D and the integrator state z is also 12D.
        We define:
          - x_dot = f(x, u) as before.
          - z_dot = x - r, where r (target state) is provided as a parameter.
        The overall state X = [x; z] is 24D.
        """
        I_xx, I_yy, I_zz = self.params["inertia"]
        mass = self.params["mass"]
        g = self.params["gravity"]

        # Define CasADi symbols.
        X = cs.MX.sym("X", 24)  # Augmented state: [x; z]
        u = cs.MX.sym("u", 6)   # Control input.
        Xdot = cs.MX.sym("Xdot", 24)
        r = cs.MX.sym("r", 12)  # Target state for x.

        # Split augmented state.
        x = X[0:12]
        z = X[12:24]

        # Unpack x.
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p_, q_, r_ = x[9], x[10], x[11]

        # Unpack control.
        Fx, Fy, Fz, tau_x, tau_y, tau_z = u[0], u[1], u[2], u[3], u[4], u[5]

        # Original dynamics.
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

        xdot = cs.vertcat(px_dot, py_dot, pz_dot,
                            vx_dot, vy_dot, vz_dot,
                            phi_dot, theta_dot, psi_dot,
                            p_dot, q_dot, r_dot)

        # Integrator dynamics.
        zdot = x - r

        # Full dynamics.
        Xdot_expl = cs.vertcat(xdot, zdot)
        f_impl = Xdot - Xdot_expl

        model = AcadosModel()
        model.f_expl_expr = Xdot_expl
        model.f_impl_expr = f_impl
        model.x = X
        model.xdot = Xdot
        model.u = u
        model.p = r  # r is the parameter for the target state.
        model.name = "augmented_hexarotor_model"

        return model

    def build_acados_ocp_solver(self):
        """
        Build the ACADOS OCP solver for the augmented hexarotor model.
        The stage cost penalizes:
          - The error in x: (x - r)
          - The integrator state z (we desire z -> 0)
          - The deviation of u from a nominal hover input.
        Terminal cost penalizes (x - r) and z.
        """
        params = self.params
        ocp = AcadosOcp()
        ocp.model = self.model

        # Horizon length.
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon

        # Set initial condition constraint for augmented state.
        ocp.constraints.x0 = np.zeros(24)

        # Cost weighting matrices.
        Q_x = np.diag([
            params['tracking_weight_pos'],
            params['tracking_weight_pos'],
            params['tracking_weight_pos'] * 25,
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
        integrator_weight = params.get("integrator_weight", 0.1)
        S_z = np.diag([integrator_weight] * 12)
        R = np.diag([
            params['control_weight'],
            params['control_weight'],
            params['control_weight'],
            params['control_weight'] * 10,
            params['control_weight'] * 10,
            params['control_weight'] * 10
        ])

        # Stage cost: residual = [ x - r; z; u - u_nom ]
        W = np.block([
            [Q_x,             np.zeros((12, 12)), np.zeros((12, 6))],
            [np.zeros((12, 12)), S_z,             np.zeros((12, 6))],
            [np.zeros((6, 12)),  np.zeros((6, 12)),   R]
        ])
        ocp.cost.W = W

        # Terminal cost: residual = [ x - r; z ]
        Q_term = np.block([
            [Q_x,           np.zeros((12, 12))],
            [np.zeros((12, 12)), S_z]
        ])
        ocp.cost.W_e = Q_term

        # Define Vx and Vu to extract the residual.
        Vx = np.block([np.eye(24), np.zeros((24, 6))])
        Vu = np.block([np.zeros((6, 24)), np.eye(6)])
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(24)

        # Set control constraints.
        max_force = params['max_force']
        max_torque = params['max_torque']
        ocp.constraints.lbu = np.array([-max_force, -max_force, 0,
                                        -max_torque, -max_torque, -max_torque])
        ocp.constraints.ubu = np.array([ max_force,  max_force, 12*max_force,
                                         max_torque,  max_torque,  max_torque])
        ocp.constraints.idxbu = np.arange(6)

        # Solver options.
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # Note: Do not call any `.set` methods on the `ocp` object here.
        # Dynamic updates (e.g. references, parameters) will be done via the solver instance.

        ocp_solver = AcadosOcpSolver(ocp)
        return ocp_solver

    def compute_control(self, state, target_state, dt):
        """
        Compute the control input using the offset-free MPC.
          - state: current 12D state measurement.
          - target_state: desired 12D reference.
          - dt: timestep.
        This function sets the initial augmented state as [state; self.int_state],
        passes the target state as parameter, and then after solving,
        updates the integrator state.
        """
        if not self.initialized:
            return np.zeros(6)

        # Build augmented initial state.
        X0 = np.concatenate([state, self.int_state])
        self.ocp_solver.set(0, "x", X0)
        self.ocp_solver.set(0, "lbx", X0)
        self.ocp_solver.set(0, "ubx", X0)

        # Update the parameter (target state) at each stage.
        for i in range(self.horizon + 1):
            self.ocp_solver.set(i, "p", target_state)

        # Compute nominal hover input (for use in the cost residual).
        phi, theta, psi = target_state[6], target_state[7], target_state[8]
        Fx_nom = -self.params["mass"] * self.params["gravity"] * np.sin(theta)
        Fy_nom =  self.params["mass"] * self.params["gravity"] * np.sin(phi) * np.cos(theta)
        Fz_nom =  self.params["mass"] * self.params["gravity"] * np.cos(phi) * np.cos(theta)
        u_nom = np.array([Fx_nom, Fy_nom, Fz_nom, 0.0, 0.0, 0.0])
        yref = np.concatenate([np.zeros(12), np.zeros(12), u_nom])
        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            self.ocp_solver.set(i, "u", self.previous_control)
        self.ocp_solver.set(self.horizon, "yref", np.zeros(24))

        # Solve the OCP.
        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[ACADOS] solver returned status {status}, something went wrong.")
            return self.previous_control

        # Extract the first control input.
        u_opt = np.array(self.ocp_solver.get(0, "u"))
        self.previous_control = u_opt.copy()

        # Update the integrator state.
        self.int_state = self.int_state + dt * (state - target_state)

        return u_opt

# (Simulation and main functions remain as in your original implementation.)
