#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <acado_code_generation.hpp>

USING_NAMESPACE_ACADO

int main() {
    // Differential state (e.g., position)
    DifferentialState x;

    // Control input (e.g., velocity)
    Control u;

    // Target position (parameter)
    Parameter target;

    // Differential equation (simple 1D system: dot(x) = u)
    DifferentialEquation f;
    f << dot(x) == u;

    // Weight matrices for running cost and terminal cost
    DMatrix W_running(2, 2); // Running cost weights
    W_running(0, 0) = 10.0;  // Weight for state tracking
    W_running(1, 1) = 1.0;   // Weight for control effort

    DMatrix W_terminal(1, 1); // Terminal cost weights
    W_terminal(0, 0) = 20.0;  // Heavier weight for final state tracking

    // Running cost function: state tracking and control effort
    Function running_cost;
    running_cost << x - target; // State tracking error
    running_cost << u;          // Control effort

    // Terminal cost function: state tracking only
    Function terminal_cost;
    terminal_cost << x - target; // Terminal state tracking

    // Define the OCP
    OCP ocp(0.0, 1.0, 10); // Time horizon: 1.0s, 10 control intervals
    ocp.minimizeLSQ(W_running, running_cost);
    ocp.minimizeLSQEndTerm(W_terminal, terminal_cost); // Terminal cost
    ocp.subjectTo(f);
    ocp.subjectTo(-1.0 <= u <= 1.0); // Control limits

    // Export the solver
    OCPexport mpc(ocp);
    mpc.set(DISCRETIZATION_TYPE, SINGLE_SHOOTING);
    mpc.set(GENERATE_TEST_FILE, YES);
    mpc.set(GENERATE_MAKE_FILE, YES);
    mpc.set(GENERATE_MATLAB_INTERFACE, NO);
    mpc.exportCode("mpc_solver");

    return 0;
}
