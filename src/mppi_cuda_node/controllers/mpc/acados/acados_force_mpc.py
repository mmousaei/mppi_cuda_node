import casadi as cs
import numpy as np

def build_force_mpc_3d(params):
    """
    Build a separate MPC for 3D force control.

    """
    # MPC parameters for force control
    N = params.get("force_horizon", 10)      # Horizon length
    dt = params.get("force_dt", 0.05)          # Time step for force MPC
    alpha = params.get("force_alpha", 1.0)     # Damping coefficient for error
    beta = params.get("force_beta", 1.0)       # Gain on delta u
    lambda_weight = params.get("force_lambda", 0.1)  # Weight for control effort

    # For simplicity, use scalar gains and treat the dynamics elementwise.
    # Define symbolic variables for state e and control delta_u .
    e = cs.MX.sym("e", 3)  # force error state
    du = cs.MX.sym("du", 3) # force control adjustment

    # Dynamics: e_dot = -alpha e + beta delta_u.
    e_dot = -alpha * e + beta * du
    f_dyn = cs.Function("f_dyn", [e, du], [e_dot])
    
    # Decision variables: U ∈ ℝ^(3×N) and state trajectory E ∈ ℝ^(3×(N+1))
    U = cs.MX.sym("U", 3, N)
    E = cs.MX.sym("E", 3, N+1)
    
    # Parameter: initial force error e0 
    e0 = cs.MX.sym("e0", 3)
    
    # Initialize cost and constraints list
    cost = 0
    constraints = []
    
    # Constraint: initial state must equal e0.
    constraints.append(E[:, 0] - e0)
    
    # Build dynamics constraints and stage cost.
    for k in range(N):
        # Euler integration for dynamics:
        e_next = E[:, k] + dt * f_dyn(E[:, k], U[:, k])
        constraints.append(E[:, k+1] - e_next)
        
        # Stage cost: tracking error plus control effort.
        cost += cs.sumsqr(E[:, k]) + lambda_weight * cs.sumsqr(U[:, k])
    
    # Add terminal cost on the final state.
    cost += cs.sumsqr(E[:, N])
    
    # Stack all constraints into a single vector.
    constraints = cs.vertcat(*constraints)
    
    # Create a CasADi function for the force MPC OCP.
    force_mpc = cs.Function('force_mpc', 
                            [e0, U, E],  # Here we include e0 as parameter and U, E as decision variables
                            [cost, constraints],
                            ['e0', 'U', 'E'],
                            ['cost', 'constraints'])
    
    # Pack the OCP data into a dictionary for later use.
    ocp = {
        "N": N,
        "dt": dt,
        "f_dyn": f_dyn,
        "force_mpc": force_mpc,
        "e0": e0,
        "U": U,
        "E": E,
        "cost_expr": cost,
        "constraints_expr": constraints
    }
    
    return ocp

# Example usage:
if __name__ == "__main__":
    # Define parameters for force MPC
    force_mpc_params = {
        "force_horizon": 10,
        "force_dt": 0.05,
        "force_alpha": 1.0,
        "force_beta": 1.0,
        "force_lambda": 0.1
    }
    
    # Build the 3D force MPC OCP
    ocp = build_force_mpc_3d(force_mpc_params)
    
    # Suppose we have a predicted contact force F_pred and a desired contact force F_des (both 3D vectors)
    F_pred = np.array([0.0, 0.0, 8.0])   # for example, measured or estimated force (N)
    F_des = np.array([0.0, 0.0, 10.0])     # desired force (N)
    
    # Compute force error
    e0_val = F_pred - F_des  # This is our initial error in R3
    
    # Here you would solve the force MPC OCP to obtain the sequence U (and particularly the first control adjustment).
    # In practice, you would use an optimizer (like IPOPT, ACADOS, etc.) to solve:
    #   minimize cost subject to constraints = 0.
    # For demonstration, we only build the OCP function.
    sol = ocp["force_mpc"](e0=e0_val, U=np.zeros((3, ocp["N"])), E=np.zeros((3, ocp["N"]+1)))
    cost_val = sol["cost"]
    constraints_val = sol["constraints"]
    
    print("Initial force error e0:", e0_val)
    print("OCP cost (with zero initial guess):", cost_val)
    print("OCP constraint residuals:", constraints_val)
    
    # In a real implementation, after solving, you would extract Δu[0] (the first control adjustment)
    # and then add it to your nominal control computed by your trajectory MPC:
    #   u_total = u_nom + Δu_first
