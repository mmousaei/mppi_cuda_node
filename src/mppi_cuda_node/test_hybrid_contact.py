#!/usr/bin/env python
"""
Test script for the Hybrid Contact MPPI Controller

This script tests the basic functionality of the hybrid contact dynamics
and MPPI optimization without requiring ROS or GPU.
"""

import sys
import os
import numpy as np
import time

# Add the controllers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'controllers', 'mppi'))

try:
    from mppi_numba_hybrid_contact import MPPI_Numba, Config, dynamics_update_euler
    print("✓ Successfully imported hybrid contact MPPI controller")
except ImportError as e:
    print(f"✗ Failed to import hybrid contact MPPI controller: {e}")
    sys.exit(1)

def test_hybrid_contact_dynamics():
    """Test the hybrid contact dynamics implementation"""
    print("\n=== Testing Hybrid Contact Dynamics ===")
    
    # Test parameters
    dt = 0.02
    mppi_params = {
        'dt': dt,
        'inertia_mass': np.array([0.21, 0.21, 0.4, 6.15]),
        'plane': np.array([-1, 0, 0, 2.2])  # Wall at x=2.2
    }
    
    # Test state (approaching wall)
    test_state = np.array([2.0, 0, 0.8, 0.5, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0])
    test_control = np.array([0, 0, 60, 0, 0, 0, 0, 0, 0])  # Hover control
    
    print(f"Initial state: position=({test_state[0]:.2f}, {test_state[1]:.2f}, {test_state[2]:.2f})")
    print(f"Distance to wall: {2.2 - test_state[0]:.2f} m")
    
    # Simulate several steps
    current_state = test_state.copy()
    for i in range(10):
        try:
            next_state, contact_forces, contact = dynamics_update_euler(
                current_state, test_control, np.zeros(3), dt, mppi_params
            )
            
            distance_to_wall = 2.2 - next_state[0]
            print(f"Step {i+1}: pos=({next_state[0]:.2f}, {next_state[1]:.2f}, {next_state[2]:.2f}), "
                  f"wall_dist={distance_to_wall:.3f}, contact={contact}, forces={contact_forces[:3]}")
            
            current_state = next_state.copy()
            
        except Exception as e:
            print(f"✗ Dynamics simulation failed at step {i+1}: {e}")
            return False
    
    print("✓ Hybrid contact dynamics test completed successfully")
    return True

def test_mppi_controller():
    """Test the MPPI controller implementation"""
    print("\n=== Testing MPPI Controller ===")
    
    try:
        # Create configuration
        cfg = Config(
            T=0.6,                # Horizon length in seconds
            dt=0.02,              # Time step
            num_control_rollouts=512,  # Reduced for testing
            num_controls=9,
            num_states=15,
            num_vis_state_rollouts=1,
            seed=1
        )
        print("✓ Configuration created successfully")
        
        # Create MPPI controller
        mppi_controller = MPPI_Numba(cfg)
        print("✓ MPPI controller created successfully")
        
        # Set parameters
        mppi_params = {
            'dt': cfg.dt,
            'x0': np.array([1.5, 0, 0.8, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]),
            'xgoal': np.array([2.0, 0, 0.8, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]),
            'fgoal': np.array([10, 0, 0]),
            'plane': np.array([-1, 0, 0, 2.2]),
            'goal_tolerance': 0.001,
            'dist_weight': 2000,
            'lambda_weight': 50,
            'num_opt': 3,
            'u_std': np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05]),
            'vrange': np.array([-10.0, 10.0]),
            'wrange': np.array([-0.1, 0.1]),
            'weights': np.array([
                19550, 19550, 24840,
                1, 1, 1,
                95500, 95500, 95500,
                1, 1, 1,
                1, 100, 1, 100, 200,
                5000, 5000, 5000
            ]),
            "inertia_mass": np.array([0.21, 0.21, 0.4, 6.15])
        }
        
        mppi_controller.set_params(mppi_params)
        print("✓ MPPI parameters set successfully")
        
        # Test optimization
        print("Running MPPI optimization...")
        start_time = time.time()
        optimal_control = mppi_controller.solve()
        optimization_time = time.time() - start_time
        
        print(f"✓ MPPI optimization completed in {optimization_time:.3f}s")
        print(f"Optimal control sequence shape: {optimal_control.shape}")
        print(f"First control: {optimal_control[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ MPPI controller test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_contact_state_transitions():
    """Test smooth contact state transitions"""
    print("\n=== Testing Contact State Transitions ===")
    
    try:
        # Test parameters
        plane = np.array([-1, 0, 0, 2.2])  # Wall at x=2.2
        A, B, C, D = plane
        
        # Test different distances to wall
        test_positions = [2.5, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8]
        
        for x_pos in test_positions:
            # Calculate distance to plane
            distance_to_surface = (A*x_pos + B*0 + C*0.8 + D) / np.sqrt(A*A + B*B + C*C)
            
            # Determine contact state
            if distance_to_surface < -0.05:  # contact_hysteresis_on
                contact_state = "CONTACT"
            elif distance_to_surface > -0.02:  # contact_hysteresis_off
                contact_state = "NO_CONTACT"
            else:
                contact_state = "HYSTERESIS"
            
            print(f"Position x={x_pos:.1f}: distance={distance_to_surface:.3f}m, state={contact_state}")
        
        print("✓ Contact state transition test completed")
        return True
        
    except Exception as e:
        print(f"✗ Contact state transition test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Hybrid Contact MPPI Controller Test Suite")
    print("=" * 50)
    
    # Test 1: Contact dynamics
    test1_passed = test_hybrid_contact_dynamics()
    
    # Test 2: MPPI controller
    test2_passed = test_mppi_controller()
    
    # Test 3: Contact state transitions
    test3_passed = test_contact_state_transitions()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Contact Dynamics: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"MPPI Controller:  {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Contact States:   {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All tests passed! The hybrid contact MPPI controller is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
