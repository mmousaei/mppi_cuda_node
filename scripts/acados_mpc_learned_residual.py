#!/usr/bin/env python
"""
Self-contained example demonstrating:
1) A "wrong" nominal MPC (mass/inertia mismatch).
2) Data collection from the real system (with correct mass/inertia).
3) Training a small NN to learn the residual.
4) Building a second MPC that uses (nominal + NN residual).
5) Demonstrating improved performance in the second run.

Dependencies:
  - casadi
  - acados_template
  - matplotlib
  - torch (PyTorch)
"""
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

###############################################################################
# Part A: A minimal "wrong" nominal hexarotor MPC
###############################################################################
class OneStepMPCWrong:
    """
    Nominal MPC with intentionally wrong mass & inertia. We do a single-step OCP
    or short horizon each iteration, but for illustration we just keep it simple.
    """
    def __init__(self, params_wrong):
        """
        params_wrong: dictionary with 'mass' and 'inertia' that are intentionally off
        from the real system's actual mass/inertia.
        """
        self.params = params_wrong
        self.dt = params_wrong["dt"]
        self.horizon = 5  # small horizon
        self.model = self._build_hex_model_wrong()
        self.ocp_solver = self._build_acados_solver()
        self.initialized = True

    def _build_hex_model_wrong(self):
        I_xx, I_yy, I_zz = self.params["inertia"]
        mass = self.params["mass"]
        g    = self.params["gravity"]

        x = cs.MX.sym("x", 12)   # state
        u = cs.MX.sym("u", 6)    # control
        xdot = cs.MX.sym("xdot", 12)

        # unpack state
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p_, q_, r_ = x[9], x[10], x[11]

        # controls
        Fx, Fy, Fz = u[0], u[1], u[2]
        tx, ty, tz = u[3], u[4], u[5]

        # nominal dynamics with "wrong" mass
        px_dot = vx
        py_dot = vy
        pz_dot = vz

        vx_dot = (Fx / mass) - g*(cs.cos(phi)*cs.sin(theta)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi))
        vy_dot = (Fy / mass) - g*(cs.cos(phi)*cs.sin(theta)*cs.sin(psi) - cs.sin(phi)*cs.cos(psi))
        vz_dot = (Fz / mass) - g*(cs.cos(phi)*cs.cos(theta))

        phi_dot   = p_ + q_*cs.sin(phi)*cs.tan(theta) + r_*cs.cos(phi)*cs.tan(theta)
        theta_dot = q_*cs.cos(phi) - r_*cs.sin(phi)
        psi_dot   = q_*cs.sin(phi)/cs.cos(theta) + r_*cs.cos(phi)/cs.cos(theta)

        p_dot = (1.0/I_xx) * (tx + (I_yy - I_zz)*q_*r_)
        q_dot = (1.0/I_yy) * (ty + (I_zz - I_xx)*p_*r_)
        r_dot = (1.0/I_zz) * (tz + (I_xx - I_yy)*p_*q_)

        f_expl = cs.vertcat(px_dot, py_dot, pz_dot,
                            vx_dot, vy_dot, vz_dot,
                            phi_dot, theta_dot, psi_dot,
                            p_dot, q_dot, r_dot)
        f_impl = xdot - f_expl

        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "hex_wrong"
        return model

    def _build_acados_solver(self):
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon
        ocp.constraints.x0 = np.zeros(12)

        p = self.params
        Q = np.diag([
            p['w_pos'] , p['w_pos'] , p['w_pos']*16,
            p['w_vel'] , p['w_vel'] , p['w_vel'],
            p['w_att'] , p['w_att'] , p['w_att'],
            p['w_angv'], p['w_angv'], p['w_angv']
        ])
        R = np.diag([
            p['w_ctrl'], p['w_ctrl'], p['w_ctrl'],
            p['w_ctrl']*10, p['w_ctrl']*10, p['w_ctrl']*10
        ])
        W = np.block([
            [Q, np.zeros((12,6))],
            [np.zeros((6,12)), R]
        ])
        ocp.cost.W = W
        ocp.cost.Vx = np.eye(18)[:, :12]
        ocp.cost.Vu = np.eye(18)[:, 12:]
        ocp.cost.yref = np.zeros(18)

        maxF = p['maxF']
        maxT = p['maxT']
        ocp.constraints.lbu = np.array([-maxF, -maxF, 0, -maxT, -maxT, -maxT])
        ocp.constraints.ubu = np.array([ maxF,  maxF, 12*maxF, maxT, maxT, maxT])
        ocp.constraints.idxbu = np.arange(6)

        ocp.solver_options.qp_solver        = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx   = 'EXACT'   # or 
        ocp.solver_options.sim_method_jacobian = 'FINITE_DIFFERENCE'
        ocp.solver_options.integrator_type  = 'ERK'
        ocp.solver_options.nlp_solver_type  = 'SQP_RTI'

        return AcadosOcpSolver(ocp)

    def compute_control(self, x_curr, x_ref, u_guess=None):
        if not self.initialized:
            return np.zeros(6)
        if u_guess is None:
            u_guess = np.zeros(6)
        self.ocp_solver.set(0, "lbx", x_curr)
        self.ocp_solver.set(0, "ubx", x_curr)
        self.ocp_solver.set(0, "x", x_curr)

        # build yref
        # for simplicity, we guess we want a "hover" in Fz. just do:
        hover_u = np.array([0,0,self.params["mass"]*self.params["gravity"],0,0,0])
        yref = np.concatenate([x_ref, hover_u])

        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            self.ocp_solver.set(i, "u", u_guess)

        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[OneStepMPCWrong] solver returned {status}. Infeasible?")

        u_opt = np.array(self.ocp_solver.get(0, "u"))
        return u_opt


###############################################################################
# Part B: "Real" dynamics with correct mass/inertia + data collection
###############################################################################
def hex_dynamics_real(x, u, p):
    """
    The real system dynamics, with correct mass and inertia.
    x: 12D
    u: 6D
    p: dictionary with 'mass_true', 'I_xx_true', etc.
    We'll do standard Euler equations but with correct values.
    """
    # unpack
    px, py, pz = x[0], x[1], x[2]
    vx, vy, vz = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p_, q_, r_ = x[9], x[10], x[11]

    Fx, Fy, Fz = u[0], u[1], u[2]
    tx, ty, tz = u[3], u[4], u[5]

    Ixx, Iyy, Izz = p["inertia_true"]
    mass_true = p["mass_true"]
    g = p["gravity_true"]

    px_dot = vx
    py_dot = vy
    pz_dot = vz

    vx_dot = (Fx / mass_true) - g*(np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi))
    vy_dot = (Fy / mass_true) - g*(np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))
    vz_dot = (Fz / mass_true) - g*(np.cos(phi)*np.cos(theta))

    phi_dot   = p_ + q_*np.sin(phi)*np.tan(theta) + r_*np.cos(phi)*np.tan(theta)
    theta_dot = q_*np.cos(phi) - r_*np.sin(phi)
    psi_dot   = q_*np.sin(phi)/np.cos(theta) + r_*np.cos(phi)/np.cos(theta)

    p_dot = (1.0/Ixx)*(tx + (Iyy - Izz)*q_*r_)
    q_dot = (1.0/Iyy)*(ty + (Izz - Ixx)*p_*r_)
    r_dot = (1.0/Izz)*(tz + (Ixx - Iyy)*p_*q_)

    return np.array([
        px_dot, py_dot, pz_dot,
        vx_dot, vy_dot, vz_dot,
        phi_dot, theta_dot, psi_dot,
        p_dot, q_dot, r_dot
    ], dtype=float)

def simulate_nominal_mpc_collect_data(mpc_wrong, real_params, x0, x_ref, steps=300, dt_sim=0.01):
    """
    1) We apply control from the "wrong" nominal MPC each iteration.
    2) We integrate the "real" dynamics with correct mass/inertia.
    3) We store [x, u, xdot_real - xdot_nominal] for training later.
    """
    state_log = []
    ctrl_log  = []
    resid_log = []
    x = x0.copy()

    def nominal_hex_dynamics_wrong(x_, u_):
        # Re-implement the 'wrong' nominal formula to get xdot_nom
        I_xx, I_yy, I_zz = mpc_wrong.params["inertia"]
        mass = mpc_wrong.params["mass"]
        g = mpc_wrong.params["gravity"]
        px_, py_, pz_ = x_[0], x_[1], x_[2]
        vx_, vy_, vz_ = x_[3], x_[4], x_[5]
        phi_, th_, ps_ = x_[6], x_[7], x_[8]
        p__, q__, r__  = x_[9], x_[10], x_[11]
        Fx_, Fy_, Fz_ = u_[0], u_[1], u_[2]
        tx_, ty_, tz_ = u_[3], u_[4], u_[5]

        px_dot_ = vx_
        py_dot_ = vy_
        pz_dot_ = vz_
        vx_dot_ = (Fx_ / mass) - g*(np.cos(phi_)*np.sin(th_)*np.cos(ps_) + np.sin(phi_)*np.sin(ps_))
        vy_dot_ = (Fy_ / mass) - g*(np.cos(phi_)*np.sin(th_)*np.sin(ps_) - np.sin(phi_)*np.cos(ps_))
        vz_dot_ = (Fz_ / mass) - g*(np.cos(phi_)*np.cos(th_))
        phi_dot_   = p__ + q__*np.sin(phi_)*np.tan(th_) + r__*np.cos(phi_)*np.tan(th_)
        th_dot_    = q__*np.cos(phi_) - r__*np.sin(phi_)
        ps_dot_    = q__*np.sin(phi_)/np.cos(th_) + r__*np.cos(phi_)/np.cos(th_)
        p_dot_ = (1.0/I_xx)*(tx_ + (I_yy - I_zz)*q__*r__)
        q_dot_ = (1.0/I_yy)*(ty_ + (I_zz - I_xx)*p__*r__)
        r_dot_ = (1.0/I_zz)*(tz_ + (I_xx - I_yy)*p__*q__)
        return np.array([
            px_dot_, py_dot_, pz_dot_,
            vx_dot_, vy_dot_, vz_dot_,
            phi_dot_, th_dot_, ps_dot_,
            p_dot_, q_dot_, r_dot_
        ])

    for k in range(steps):
        # compute control
        u_k = mpc_wrong.compute_control(x, x_ref)

        # store
        ctrl_log.append(u_k.copy())
        state_log.append(x.copy())

        # real derivative
        xdot_real = hex_dynamics_real(x, u_k, real_params)
        # nominal derivative from wrong model
        xdot_nom  = nominal_hex_dynamics_wrong(x, u_k)
        residual  = xdot_real - xdot_nom
        resid_log.append(residual.copy())

        # euler integration
        x = x + dt_sim * xdot_real

    return np.array(state_log), np.array(ctrl_log), np.array(resid_log)


###############################################################################
# Part C: Train a small NN on residual data
###############################################################################
class ResidualMLP(nn.Module):
    def __init__(self, in_dim=18, out_dim=12, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_residual_nn(X_data, Y_data, epochs=50, lr=1e-3):
    """
    X_data: shape (N, 18)  [ x(12) + u(6) ]
    Y_data: shape (N, 12)  [ residual 12D ]
    """
    model = ResidualMLP(in_dim=18, out_dim=12, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    N = X_data.shape[0]
    batch_size = 128

    X_tensor = torch.from_numpy(X_data.astype(np.float32))
    Y_tensor = torch.from_numpy(Y_data.astype(np.float32))

    for ep in range(epochs):
        perm = np.random.permutation(N)
        X_tensor = X_tensor[perm]
        Y_tensor = Y_tensor[perm]
        for i in range(0, N, batch_size):
            Xb = X_tensor[i:i+batch_size]
            Yb = Y_tensor[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            optimizer.step()
        if (ep+1)%10==0:
            print(f"Epoch {ep+1}, Loss={loss.item():.6f}")
    return model


###############################################################################
# Part D: Export the trained model to ONNX, then create a symbolic MPC
###############################################################################
import onnx
import onnx2casadi

def export_mlp_to_onnx(mlp_model, onnx_filename="my_mlp.onnx"):
    """
    mlp_model : trained PyTorch model, input shape (1,18), output shape (1,12).
    onnx_filename: path to save the ONNX file.
    """
    mlp_model.eval()
    dummy_input = torch.randn(1, 18, dtype=torch.float32)
    torch.onnx.export(
        mlp_model,
        dummy_input,
        onnx_filename,
        input_names=["input_0"],   # we'll reference these in onnx2casadi
        output_names=["output_0"],
        opset_version=11
    )
    print(f"Exported PyTorch MLP to ONNX: '{onnx_filename}'")


class OneStepMPCWithNNOnnx:
    """
    Instead of a Python callback for the NN residual, we parse the ONNX file
    into a symbolic CasADi function and combine it with the nominal model.
    CasADi can do normal AD on this function, so no derivative error occurs.
    """
    def __init__(self, params_nom, onnx_path):
        """
        params_nom: dictionary with nominal mass/inertia
        onnx_path : the file path of the .onnx model, e.g. "my_mlp.onnx"
                    which has input name "input_0" and output name "output_0".
        """
        self.params = params_nom
        self.dt = params_nom["dt"]
        self.horizon = 5

        # 1) Build a CasADi expression from the ONNX file
        self.nn_cas_func = self._load_onnx_as_casadi(onnx_path)

        # 2) Build the combined hexarotor + residual model
        self.model = self._build_hex_model_with_onnx()
        # 3) Construct the solver
        self.ocp_solver = self._build_acados_solver()
        self.initialized = True

    def _load_onnx_as_casadi(self, onnx_path):
        # Load the model from ONNX
        onnx_model = onnx.load(onnx_path)
        # Convert to a CasADi Function. The default input name "input_0" 
        # and output name "output_0" must match your ONNX graph.
        nn_cas = onnx2casadi.convertToCasadiFunc(
            onnx_model,
            ["input_0"],   # list of input node names
            ["output_0"],  # list of output node names
            use_numpy=False
        )
        return nn_cas

    def _build_hex_model_with_onnx(self):
        """
        Same nominal eqns, then add the ONNX-based residual as a CasADi expression.
        """
        I_xx, I_yy, I_zz = self.params["inertia"]
        mass = self.params["mass"]
        g    = self.params["gravity"]

        # CasADi variables
        x = cs.MX.sym("x", 12)
        u = cs.MX.sym("u", 6)
        xdot = cs.MX.sym("xdot", 12)

        # Nominal part
        px_dot = x[3]
        py_dot = x[4]
        pz_dot = x[5]

        vx_dot = (u[0]/mass) - g*(cs.cos(x[6])*cs.sin(x[7])*cs.cos(x[8]) + cs.sin(x[6])*cs.sin(x[8]))
        vy_dot = (u[1]/mass) - g*(cs.cos(x[6])*cs.sin(x[7])*cs.sin(x[8]) - cs.sin(x[6])*cs.cos(x[8]))
        vz_dot = (u[2]/mass) - g*(cs.cos(x[6])*cs.cos(x[7]))

        phi_dot   = x[9] + x[10]*cs.sin(x[6])*cs.tan(x[7]) + x[11]*cs.cos(x[6])*cs.tan(x[7])
        theta_dot = x[10]*cs.cos(x[6]) - x[11]*cs.sin(x[6])
        psi_dot   = x[10]*cs.sin(x[6])/cs.cos(x[7]) + x[11]*cs.cos(x[6])/cs.cos(x[7])

        p_dot = (1.0/I_xx)*(u[3] + (I_yy - I_zz)*x[10]*x[11])
        q_dot = (1.0/I_yy)*(u[4] + (I_zz - I_xx)*x[9]*x[11])
        r_dot = (1.0/I_zz)*(u[5] + (I_xx - I_yy)*x[9]*x[10])

        f_nom = cs.vertcat(px_dot, py_dot, pz_dot,
                           vx_dot, vy_dot, vz_dot,
                           phi_dot, theta_dot, psi_dot,
                           p_dot, q_dot, r_dot)

        # 2) ONNX-based residual
        xu = cs.vertcat(x, u)  # shape(18,)
        f_res = self.nn_cas_func(xu)  # returns 12D CasADi expression

        f_expl = f_nom + f_res
        f_impl = xdot - f_expl

        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "hex_with_onnx_nn"
        return model

    def _build_acados_solver(self):
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.dt*self.horizon
        ocp.constraints.x0 = np.zeros(12)

        p = self.params
        Q = np.diag([p['w_pos'], p['w_pos'], p['w_pos']*16,
                     p['w_vel'], p['w_vel'], p['w_vel'],
                     p['w_att'], p['w_att'], p['w_att'],
                     p['w_angv'], p['w_angv'], p['w_angv']])
        R = np.diag([p['w_ctrl'], p['w_ctrl'], p['w_ctrl'],
                     p['w_ctrl']*10, p['w_ctrl']*10, p['w_ctrl']*10])
        W = np.block([
            [Q, np.zeros((12,6))],
            [np.zeros((6,12)), R]
        ])
        ocp.cost.W = W
        ocp.cost.Vx = np.eye(18)[:, :12]
        ocp.cost.Vu = np.eye(18)[:, 12:]
        ocp.cost.yref = np.zeros(18)

        maxF = p['maxF']
        maxT = p['maxT']
        ocp.constraints.lbu = np.array([-maxF, -maxF, 0, -maxT, -maxT, -maxT])
        ocp.constraints.ubu = np.array([ maxF,  maxF, 12*maxF, maxT, maxT, maxT])
        ocp.constraints.idxbu = np.arange(6)

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        return AcadosOcpSolver(ocp)

    def compute_control(self, x_curr, x_ref, u_guess=None):
        if not self.initialized:
            return np.zeros(6)
        if u_guess is None:
            u_guess = np.zeros(6)
        self.ocp_solver.set(0, "lbx", x_curr)
        self.ocp_solver.set(0, "ubx", x_curr)
        self.ocp_solver.set(0, "x", x_curr)

        mass = self.params["mass"]
        g = self.params["gravity"]
        hover_u = np.array([0,0,mass*g, 0,0,0])
        yref = np.concatenate([x_ref, hover_u])
        for i in range(self.horizon):
            self.ocp_solver.set(i, "yref", yref)
            self.ocp_solver.set(i, "u", u_guess)

        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[OneStepMPCWithNNOnnx] solver status = {status}. Check feasibility")

        return np.array(self.ocp_solver.get(0,"u"))


def simulate_mpc_with_onnx_nn(mpc, real_params, x0, x_ref, steps=300, dt_sim=0.01):
    """
    Same as simulate_mpc_with_nn, but we rename for clarity. 
    This uses the ONNX-based symbolic residual in the OCP,
    while the real system is integrated with correct mass/inertia.
    """
    state_log = []
    ctrl_log  = []
    x = x0.copy()

    for k in range(steps):
        u_k = mpc.compute_control(x, x_ref)
        ctrl_log.append(u_k.copy())
        state_log.append(x.copy())
        xdot_real = hex_dynamics_real(x, u_k, real_params)
        x = x + dt_sim*xdot_real

    return np.array(state_log), np.array(ctrl_log)


###############################################################################
# REPLACE this final section in your main() after training the MLP
###############################################################################
def main():
    # (A) Setup real & nominal params (same as your code)...

    # (B) Build "wrong" MPC & simulate -> (X_data, Y_data)...

    # (C) Train the MLP
    mlp_model = train_residual_nn(X_data, Y_data, epochs=50, lr=1e-3)

    # (D) **Export** the MLP to ONNX
    onnx_filename = "my_mlp.onnx"
    mlp_model.eval()
    dummy_input = torch.randn(1, 18, dtype=torch.float32)
    torch.onnx.export(
        mlp_model, 
        dummy_input,
        onnx_filename,
        input_names=["input_0"], 
        output_names=["output_0"], 
        opset_version=11
    )
    print(f"Exported MLP to ONNX => {onnx_filename}")

    # (E) Build second MPC using the onnx2casadi approach
    mpc_nn_onnx = OneStepMPCWithNNOnnx(wrong_params, onnx_filename)

    # (F) Simulate the new approach
    x_log_nn, u_log_nn = simulate_mpc_with_onnx_nn(
        mpc_nn_onnx,
        {
            "inertia_true": real_params["inertia_true"],
            "mass_true": real_params["mass_true"],
            "gravity_true": real_params["gravity_true"]
        },
        x0, x_ref, steps=steps, dt_sim=dt_sim
    )

    # (G) compare/plot x_log_wrong vs x_log_nn
    ...



###############################################################################
# Main demonstration
###############################################################################
def main():
    # A) Setup params
    # Real system
    real_params = {
        "mass_true": 10.0,                   # correct mass is 10
        "inertia_true": [0.12, 0.12, 0.24],  # correct inertia
        "gravity_true": 9.81
    }
    # Wrong nominal MPC params
    wrong_params = {
        "mass": 7.0,                         # intentionally off
        "inertia": [0.05, 0.05, 0.13],     # also slightly off
        "gravity": 9.81,
        "dt": 0.3,
        # cost
        "w_pos": 10.0,
        "w_vel": 3.0,
        "w_att": 80.0,
        "w_angv":50.0,
        "w_ctrl":0.005,
        # constraints
        "maxF": 20.0,
        "maxT": 0.2
    }
    # B) Build nominal "wrong" MPC
    mpc_wrong = OneStepMPCWrong(wrong_params)

    # C) simulate + collect data
    x0 = np.zeros(12)
    x_ref = np.array([1.0, 0.0, 2.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0])
    steps = 400
    dt_sim = 0.01
    x_log_wrong, u_log_wrong, resid_log = simulate_nominal_mpc_collect_data(
        mpc_wrong, {
            "inertia_true": real_params["inertia_true"],
            "mass_true": real_params["mass_true"],
            "gravity_true": real_params["gravity_true"]
        }, x0, x_ref, steps=steps, dt_sim=dt_sim
    )
    print("Data collection done. Now shape of x_log_wrong:", x_log_wrong.shape)

    # Build X_data = [x(12) + u(6)] = 18D, Y_data = residual(12D)
    X_data = []
    Y_data = []
    for i in range(steps):
        x_ = x_log_wrong[i]
        u_ = u_log_wrong[i]
        X_data.append( np.hstack([x_, u_]) )
        Y_data.append( resid_log[i] )
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    print("X_data shape:", X_data.shape, "Y_data shape:", Y_data.shape)

    # (D) Train the MLP
    mlp_model = train_residual_nn(X_data, Y_data, epochs=50, lr=1e-3)

    # (E) **Export** the MLP to ONNX
    onnx_filename = "my_mlp.onnx"
    mlp_model.eval()
    dummy_input = torch.randn(1, 18, dtype=torch.float32)
    torch.onnx.export(
        mlp_model, 
        dummy_input,
        onnx_filename,
        input_names=["input_0"], 
        output_names=["output_0"], 
        opset_version=11
    )
    print(f"Exported MLP to ONNX => {onnx_filename}")

    # (F) Build second MPC using the onnx2casadi approach
    mpc_nn_onnx = OneStepMPCWithNNOnnx(wrong_params, onnx_filename)

    # (G) Simulate the new approach
    x_log_nn, u_log_nn = simulate_mpc_with_onnx_nn(
        mpc_nn_onnx,
        {
            "inertia_true": real_params["inertia_true"],
            "mass_true": real_params["mass_true"],
            "gravity_true": real_params["gravity_true"]
        },
        x0, x_ref, steps=steps, dt_sim=dt_sim
    )
    print("Second simulation done with NN residual included. Compare results.")

    # Plot results
    t = np.arange(steps)*dt_sim
    plt.figure()
    plt.plot(t, x_log_wrong[:steps,0], label='px (wrong MPC)')
    plt.plot(t, x_log_nn[:steps,0],   label='px (NN MPC)')
    plt.title("Position X")
    plt.legend(); plt.grid()

    plt.figure()
    plt.plot(t, x_log_wrong[:steps,2], label='pz (wrong MPC)')
    plt.plot(t, x_log_nn[:steps,2],   label='pz (NN MPC)')
    plt.title("Position Z")
    plt.legend(); plt.grid()

    plt.show()


if __name__=="__main__":
    main()
