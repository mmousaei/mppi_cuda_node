#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>
#include <acado_code_generation.hpp>

USING_NAMESPACE_ACADO

int main() {
    // Differential states: x[0:2] -> position, x[3:5] -> velocity, x[6:8] -> Euler angles, x[9:11] -> angular velocity
    DifferentialState x[12];

    // Control inputs: u[0:2] -> forces, u[3:5] -> torques
    Control u[6];

    // Target states as parameters
    Parameter target[12];

    // Constants
    const double mass = 3.49;
    const double g = 9.81;
    const double I_xx = 0.115125971;
    const double I_yy = 0.116524229;
    const double I_zz = 0.230387752;

    // Hexarotor dynamics
    DifferentialEquation f;

    // Translational dynamics
    f << dot(x[0]) == x[3];
    f << dot(x[1]) == x[4];
    f << dot(x[2]) == x[5];
    f << dot(x[3]) == (1.0 / mass) * u[0] - g * (cos(x[6]) * sin(x[7]) * cos(x[8]) + sin(x[6]) * sin(x[8]));
    f << dot(x[4]) == (1.0 / mass) * u[1] - g * (cos(x[6]) * sin(x[7]) * sin(x[8]) - sin(x[6]) * cos(x[8]));
    f << dot(x[5]) == (1.0 / mass) * u[2] - g * (cos(x[6]) * cos(x[7]));

    // Rotational dynamics
    f << dot(x[6]) == x[9] + x[10] * sin(x[6]) * tan(x[7]) + x[11] * cos(x[6]) * tan(x[7]);
    f << dot(x[7]) == x[10] * cos(x[6]) - x[11] * sin(x[6]);
    f << dot(x[8]) == x[10] * sin(x[6]) / cos(x[7]) + x[11] * cos(x[6]) / cos(x[7]);
    f << dot(x[9]) == (1.0 / I_xx) * (u[3] + (I_yy - I_zz) * x[10] * x[11]);
    f << dot(x[10]) == (1.0 / I_yy) * (u[4] + (I_zz - I_xx) * x[9] * x[11]);
    f << dot(x[11]) == (1.0 / I_zz) * (u[5] + (I_xx - I_yy) * x[9] * x[10]);

    // Cost weights
    const double pose_weight = 10.0;            // Position tracking
    const double velocity_weight = 5.0;        // Velocity tracking
    const double attitude_weight = 8.0;        // Attitude tracking
    const double angular_velocity_weight = 3.0; // Angular velocity tracking
    const double control_weight = 0.1;         // Penalize control effort

    // Running cost
    Function running_cost;
    running_cost << pose_weight * (x[0] - target[0]);
    running_cost << pose_weight * (x[1] - target[1]);
    running_cost << pose_weight * (x[2] - target[2]);
    running_cost << velocity_weight * (x[3] - target[3]);
    running_cost << velocity_weight * (x[4] - target[4]);
    running_cost << velocity_weight * (x[5] - target[5]);
    running_cost << attitude_weight * (x[6] - target[6]);
    running_cost << attitude_weight * (x[7] - target[7]);
    running_cost << attitude_weight * (x[8] - target[8]);
    running_cost << angular_velocity_weight * (x[9] - target[9]);
    running_cost << angular_velocity_weight * (x[10] - target[10]);
    running_cost << angular_velocity_weight * (x[11] - target[11]);
    for (int i = 0; i < 6; ++i) {
        running_cost << control_weight * u[i];
    }

    // Terminal cost
    Function terminal_cost;
    terminal_cost << pose_weight * (x[0] - target[0]);
    terminal_cost << pose_weight * (x[1] - target[1]);
    terminal_cost << pose_weight * (x[2] - target[2]);

    // Weight matrices for running and terminal costs
    DMatrix W_running(running_cost.getDim(), running_cost.getDim());
    W_running.setIdentity();
    DMatrix W_terminal(terminal_cost.getDim(), terminal_cost.getDim());
    W_terminal.setIdentity();

    // Define the OCP
    OCP ocp(0.0, 1.0, 10); // Time horizon: 1.0s, 10 control intervals
    ocp.minimizeLSQ(W_running, running_cost);
    ocp.minimizeLSQEndTerm(W_terminal, terminal_cost);
    ocp.subjectTo(f);

    // Control constraints
    ocp.subjectTo(-60.0 <= u[0] <= 60.0);
    ocp.subjectTo(-60.0 <= u[1] <= 60.0);
    ocp.subjectTo(0.0 <= u[2] <= 60.0); // Thrust is positive
    ocp.subjectTo(-10.0 <= u[3] <= 10.0);
    ocp.subjectTo(-10.0 <= u[4] <= 10.0);
    ocp.subjectTo(-10.0 <= u[5] <= 10.0);

    // Export the solver
    OCPexport mpc(ocp);
    mpc.set(DISCRETIZATION_TYPE, SINGLE_SHOOTING);
    mpc.set(GENERATE_TEST_FILE, YES);
    mpc.set(GENERATE_MAKE_FILE, YES);
    mpc.set(GENERATE_MATLAB_INTERFACE, NO);
    mpc.set(USE_INITIAL_GUESS, YES); // Enable initial guess

    // Set an example initial guess for control
    DVector u_init(6);
    u_init.setAll(0.0); // Set initial guess to zero
    mpc.setInit("u", u_init);


    mpc.exportCode("mpc_solver");

    return 0;
}


// To generate Acado solver do:
// g++ mpc_ocp.cpp -o generate_solver     -I/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src/mppi_cuda_node/scripts/acado_mpc/acado/acado/     -I/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src/mppi_cuda_node/scripts/acado_mpc/acado/     -L/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src/mppi_cuda_node/scripts/acado_mpc/acado/build/lib/     -lacado_toolkit_s -w
// export LD_LIBRARY_PATH=/home/dream_reaper/workspace/aerial_manipulation_mppi_realworld/src/mppi_cuda_node/scripts/acado_mpc/acado/build/lib:$LD_LIBRARY_PATH
// ./generate_solver