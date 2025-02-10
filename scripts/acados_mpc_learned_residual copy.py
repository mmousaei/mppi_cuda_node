import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

###############################################################################
# OneStepMPC with NN-Learned Residual
###############################################################################
class OneStepMPCWithNN:
    def __init__(self, params):
        """
        Similar to your original OneStepMPC, but we'll add a neural network 
        residual into the hexarotor model. We'll call it 'NNOneStepMPC' for clarity.
        """
        self.params = params
        self.dt = params["dt"]
        self.previous_control = np.zeros(6)

        # Build the model: nominal + NN residual
        self.model = self.build_nn_hexarotor_model()

        # Set up horizon, solver, etc.
        self.horizon = 5
        self.ocp_solver = self.build_acados_ocp_solver()
        self.initialized = True

    def build_nn_hexarotor_model(self):
        """
        Build a 6-DoF hexarotor model in CasADi, 
        then add the neural network (NN) residual to the nominal dynamics.
        """
        # Extract parameters
        I_xx, I_yy, I_zz = self.params["inertia"]
        mass = self.params["mass"]
        g    = self.params["gravity"]

        # CasADi variables
        x = cs.MX.sym("x", 12)     # state
        u = cs.MX.sym("u", 6)      # control
        xdot = cs.MX.sym("xdot", 12)

        # State shorthand
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p_, q_, r_ = x[9], x[10], x[11]

        # Control shorthand
        Fx, Fy, Fz = u[0], u[1], u[2]
        tau_x, tau_y, tau_z = u[3], u[4], u[5]

        # ------------------
        # Nominal dynamics
        # ------------------
        px_dot = vx
        py_dot = vy
        pz_dot = vz

        vx_dot = (Fx / mass) - g*(cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
        vy_dot = (Fy / mass) - g*(cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
        vz_dot = (Fz / mass) - g*(cs.cos(phi)*cs.cos(theta))

        phi_dot   = p_ + q_*cs.sin(phi)*cs.tan(theta) + r_*cs.cos(phi)*cs.tan(theta)
        theta_dot = q_*cs.cos(phi) - r_*cs.sin(phi)
        psi_dot   = q_*cs.sin(phi)/cs.cos(theta) + r_*cs.cos(phi)/cs.cos(theta)

        p_dot = (1.0 / I_xx) * (tau_x + (I_yy - I_zz)*q_*r_)
        q_dot = (1.0 / I_yy) * (tau_y + (I_zz - I_xx)*p_*r_)
        r_dot = (1.0 / I_zz) * (tau_z + (I_xx - I_yy)*p_*q_)

        f_nominal = cs.vertcat(
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

        # ------------------
        # NN residual
        # ------------------
        # For demonstration, we define a small "dummy" symbolic residual function.
        # In real usage, you'd load a pre-trained NN, e.g., from ONNX -> CasADi, or
        # approximate the NN with polynomials, or do a piecewise approach.
        f_nn = self.nn_forward_dummy(x, u)

        # Combine nominal + residual
        f_expl = f_nominal + f_nn

        # Implicit form: xdot - (f_nominal + f_nn) = 0
        f_impl = xdot - f_expl

        # Build AcadosModel
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "hex_with_nn_residual"

        return model

    def nn_forward_dummy(self, x, u):
        """
        A dummy "learned residual" expression of shape (12,).
        In real code, you'd parse a trained NN or do a CasADi-based MLP.

        For demonstration, let's do something like:
           f_nn[i] = alpha * sin( A * (some function of x,u) )
        We'll keep it extremely simple just to show how you'd add it.
        """
        alpha = 0.01  # scale of the residual
        # We'll do a single scalar that depends on some subset of x and u:
        # e.g. s = px + 0.5*Fx + ...
        px = x[0]
        vx = x[3]
        Fx = u[0]
        # dummy scalar
        s = px + 0.1*vx + 0.01*Fx

        # We'll create a length-12 residual vector that depends on s
        # e.g. each component is alpha*sin( s + i ), just for demonstration.
        residual_vec = []
        for i in range(12):
            residual_vec.append(alpha * cs.sin(s + 0.1*i))

        # Return as a CasADi vector
        f_nn_vec = cs.vertcat(*residual_vec)
        return f_nn_vec

    def build_acados_ocp_solver(self):
        """
        Build the ACADOS OCP solver for the hexarotor with an NN residual.
        Rest is basically same as your original.
        """
        params = self.params
        ocp = AcadosOcp()
        ocp.model = self.model

        # Horizon length
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon

        # x0
        ocp.constraints.x0 = np.zeros(12)

        # Cost weighting
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
        ocp.constraints.ubu = np.array([max_force, max_force, 12*max_force,
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
        Same structure: fill x0, fill yref, solve. 
        Now the dynamics internally includes the NN residual.
        """
        if not self.initialized:
            return initial_guess

        if len(state) == 12:
            self.ocp_solver.set(0, "lbx", state)
            self.ocp_solver.set(0, "ubx", state)
        self.ocp_solver.set(0, "x", state)

        # Build yref
        mass = self.params["mass"]
        g    = self.params["gravity"]
        hover_guess = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])
        yref = np.concatenate([target_state, hover_guess])

        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            self.ocp_solver.set(i, "u", initial_guess)

        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[ACADOS - NNOneStepMPC] solver returned status {status} -> check feasibility.")

        u_opt = np.array(self.ocp_solver.get(0, "u"))
        return u_opt

###############################################################################
# Simulation with NN residual
###############################################################################
def simulate_hexarotor_with_nn(params, mpc, initial_state=None, target_state=None, steps=500):
    """
    Just like your original simulate function, but we emphasize
    that the model in the OCP includes the NN residual.
    For the "real" system, let's keep it nominal or add a small mismatch.

    We'll do a simple Euler integration of the same function the OCP uses 
    (which includes the dummy NN residual) so you see the effect closed-loop.
    """
    if initial_state is None:
        initial_state = np.zeros(12)
    if target_state is None:
        target_state = np.zeros(12)

    x_sym = mpc.model.x
    u_sym = mpc.model.u
    f_expl = mpc.model.f_expl_expr   # this is (f_nominal + f_nn)
    f_func = cs.Function('f_func', [x_sym, u_sym], [f_expl])

    dt_sim = params['dt'] / 100
    state = initial_state.copy()
    state_history = [state]
    control_history = []

    mass = params["mass"]
    g    = params["gravity"]
    init_guess_u = np.array([0.0, 0.0, mass*g, 0.0, 0.0, 0.0])

    for _ in range(steps):
        # Get control from MPC
        u_opt = mpc.compute_control(state, target_state, init_guess_u, dt_sim)
        control_history.append(u_opt)

        # Euler integration with the same f_func (including the NN residual)
        x_dot_val = np.array(f_func(state, u_opt)).flatten()
        state = state + dt_sim * x_dot_val

        state_history.append(state)

    return np.array(state_history), np.array(control_history)

###############################################################################
# Main demonstration
###############################################################################
def main():
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

    # Build the MPC with a dummy NN residual in the model
    mpc = OneStepMPCWithNN(params)

    # Initial and target states
    init_state = np.zeros(12)
    target_state = np.array([
        1.0, -1.0, 2.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    steps = 500
    state_traj, control_traj = simulate_hexarotor_with_nn(
        params, mpc,
        initial_state=init_state,
        target_state=target_state,
        steps=steps
    )

    # Plot
    t_states = np.linspace(0, steps*params['dt'], steps+1)
    t_ctrl   = np.linspace(0, (steps-1)*params['dt'], steps)

    # 1) position
    plt.figure()
    plt.plot(t_states, state_traj[:,0], label='x')
    plt.plot(t_states, state_traj[:,1], label='y')
    plt.plot(t_states, state_traj[:,2], label='z')
    plt.title("Position with NN Residual")
    plt.xlabel("time (s)")
    plt.ylabel("pos (m)")
    plt.legend()
    plt.grid(True)

    # 2) orientation
    roll = state_traj[:,6]*180/np.pi
    pitch= state_traj[:,7]*180/np.pi
    yaw  = state_traj[:,8]*180/np.pi
    plt.figure()
    plt.plot(t_states, roll,  label="roll (deg)")
    plt.plot(t_states, pitch, label="pitch (deg)")
    plt.plot(t_states, yaw,   label="yaw (deg)")
    plt.title("Orientation with NN Residual")
    plt.xlabel("time (s)")
    plt.ylabel("angle (deg)")
    plt.legend()
    plt.grid(True)

    # 3) forces
    plt.figure()
    plt.plot(t_ctrl, control_traj[:,0], label='Fx')
    plt.plot(t_ctrl, control_traj[:,1], label='Fy')
    plt.plot(t_ctrl, control_traj[:,2], label='Fz')
    plt.title("Forces (N)")
    plt.xlabel("time (s)")
    plt.legend()
    plt.grid(True)

    # 4) torques
    plt.figure()
    plt.plot(t_ctrl, control_traj[:,3], label='tau_x')
    plt.plot(t_ctrl, control_traj[:,4], label='tau_y')
    plt.plot(t_ctrl, control_traj[:,5], label='tau_z')
    plt.title("Torques (Nm)")
    plt.xlabel("time (s)")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
