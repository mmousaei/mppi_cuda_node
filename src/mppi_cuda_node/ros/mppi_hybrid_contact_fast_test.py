#!/usr/bin/env python3
"""
Test Version of Fast Hybrid Contact MPPI ROS Node
This version can run independently without external message dependencies
"""

import rospy
import numpy as np
import sys
import os
import time
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import PoseStamped
import threading

# Add the package to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from controllers.mppi.mppi_numba_hybrid_contact_fast import MPPI_Numba, Config

class MPPIHybridContactFastTestNode:
    """Test version of ROS node for fast hybrid contact MPPI controller"""
    
    def __init__(self):
        rospy.init_node('mppi_hybrid_contact_fast_test_node', anonymous=True)
        
        # Initialize MPPI controller
        self.init_mppi_controller()
        
        # State variables
        self.current_state = np.zeros(15, dtype=np.float32)
        self.current_control = np.zeros(9, dtype=np.float32)
        self.goal_state = np.zeros(15, dtype=np.float32)
        self.contact_active = False
        
        # Initialize state
        self.init_state()
        
        # Initialize goal state
        self.goal_state = np.array([0.0, 0.0, -1.0,  # Position
                                   0.0, 0.0, 0.0,     # Velocity
                                   0.0, 0.0, 0.0,     # Attitude
                                   0.0, 0.0, 0.0,     # Angular velocity
                                   0.0, 0.0, 0.0],    # Contact forces
                                  dtype=np.float32)
        
        # Initialize MPPI state tracking for optimization
        self.last_mppi_state = self.current_state.copy()
        
        # ROS publishers
        self.mpc_target_pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=1)
        self.control_pub = rospy.Publisher('/mppi/control', Float64MultiArray, queue_size=1)
        self.contact_status_pub = rospy.Publisher('/mppi/contact_status', Bool, queue_size=1)
        
        # Test mode: simulate state updates
        self.test_mode = True
        self.test_counter = 0
        
        rospy.loginfo("Fast Hybrid Contact MPPI Test Node initialized")
        
        # Start test loop in separate thread
        self.test_thread = threading.Thread(target=self.test_loop, daemon=True)
        self.test_thread.start()
    
    def init_mppi_controller(self):
        """Initialize the fast MPPI controller"""
        try:
            # Configuration for fast hybrid contact
            cfg = Config(
                T=1.0,                    # Horizon length
                dt=0.02,                  # Time step
                num_control_rollouts=1024, # Number of rollouts
                num_controls=9,            # 6 original + 3 force rates
                num_states=15,            # 12 original + 3 contact forces
                num_vis_state_rollouts=20,
                seed=1
            )
            
            # Initialize MPPI controller
            self.mppi_controller = MPPI_Numba(cfg)
            
            # Set controller parameters
            self.set_mppi_params()
            
            rospy.loginfo("Fast MPPI controller initialized successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize MPPI controller: {e}")
            raise
    
    def set_mppi_params(self):
        """Set MPPI controller parameters"""
        try:
            # Goal state (hover at origin)
            xgoal = np.array([0.0, 0.0, -1.0,  # Position
                             0.0, 0.0, 0.0,     # Velocity
                             0.0, 0.0, 0.0,     # Attitude
                             0.0, 0.0, 0.0,     # Angular velocity
                             0.0, 0.0, 0.0],    # Contact forces
                            dtype=np.float32)
            
            # Initial state
            x0 = np.array([0.0, 0.0, -1.0,  # Position
                           0.0, 0.0, 0.0,     # Velocity
                           0.0, 0.0, 0.0,     # Attitude
                           0.0, 0.0, 0.0,     # Angular velocity
                           0.0, 0.0, 0.0],    # Contact forces
                          dtype=np.float32)
            
            # Cost weights
            dist_weight = 10.0
            lambda_weight = 0.1
            
            # Control noise standard deviation
            u_std = np.array([2.0, 2.0, 2.0,    # Force noise
                             0.5, 0.5, 0.5,     # Torque noise
                             0.1, 0.1, 0.1],    # Force rate noise
                            dtype=np.float32)
            
            # Additional required parameters
            vrange = np.array([-10.0, 10.0], dtype=np.float32)  # Velocity range
            wrange = np.array([-5.0, 5.0], dtype=np.float32)   # Angular velocity range
            fgoal = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Force goal
            plane = np.array([-1.0, 0.0, 0.0, 2.0], dtype=np.float32)  # Contact plane
            goal_tolerance = 0.1
            inertia_mass = np.array([0.42590587, 0.3120579, 0.11511835, 1.5], dtype=np.float32)  # Ixx, Iyy, Izz, mass
            num_opt = 1  # Number of optimization iterations
            dt = 0.02  # Time step
            
            # Set parameters
            self.mppi_params = {
                'xgoal': xgoal,
                'x0': x0,
                'dist_weight': dist_weight,
                'lambda_weight': lambda_weight,
                'u_std': u_std,
                'vrange': vrange,
                'wrange': wrange,
                'fgoal': fgoal,
                'plane': plane,
                'goal_tolerance': goal_tolerance,
                'inertia_mass': inertia_mass,
                'num_opt': num_opt,
                'weights': np.array([1.0, 1.0, 1.0,  # Position weights
                                   0.5, 0.5, 0.5,   # Velocity weights
                                   0.3, 0.3, 0.3,   # Attitude weights
                                   0.2, 0.2, 0.2,   # Angular velocity weights
                                   0.1, 0.1, 0.1]), # Contact force weights
                'dt': dt
            }
            
            # Update controller parameters using the proper method
            if hasattr(self.mppi_controller, 'set_params'):
                self.mppi_controller.set_params(self.mppi_params)
            else:
                # Fallback: direct assignment
                self.mppi_controller.params = self.mppi_params
            
            rospy.loginfo("MPPI parameters set successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to set MPPI parameters: {e}")
            raise
    
    def init_state(self):
        """Initialize the current state"""
        # Start at origin with some height
        self.current_state = np.array([0.0, 0.0, -1.0,  # Position
                                      0.0, 0.0, 0.0,     # Velocity
                                      0.0, 0.0, 0.0,     # Attitude
                                      0.0, 0.0, 0.0,     # Angular velocity
                                      0.0, 0.0, 0.0],    # Contact forces
                                     dtype=np.float32)
        
        # Initialize control to hover
        self.current_control = np.array([0.0, 0.0, 9.81,  # Hover force
                                        0.0, 0.0, 0.0,    # No torque
                                        0.0, 0.0, 0.0],   # No force rates
                                       dtype=np.float32)
    
    def simulate_state_update(self):
        """Simulate state updates for testing"""
        # Simple state evolution
        dt = 0.02
        
        # Add some movement
        self.current_state[0] += dt * 0.1 * np.sin(self.test_counter * 0.1)  # Oscillating x
        self.current_state[1] += dt * 0.1 * np.cos(self.test_counter * 0.1)  # Oscillating y
        
        # Simulate contact forces
        if self.test_counter > 50:  # Start contact after 1 second
            self.current_state[12] += dt * 0.1  # Accumulate contact force
            self.current_state[13] += dt * 0.05
            self.current_state[14] += dt * 0.02
        
        self.test_counter += 1
        
        # DYNAMIC GOAL: Update goal based on current state for more interesting behavior
        self.update_dynamic_goal()
    
    def run_mppi(self):
        """Run MPPI optimization - OPTIMIZED for speed"""
        try:
            # OPTIMIZATION: Only update x0 if it changed significantly
            state_diff = np.linalg.norm(self.current_state - self.last_mppi_state)
            if state_diff > 0.01:  # Only update if state changed by >1cm
                if hasattr(self.mppi_controller, 'set_params'):
                    # Create a copy of params and update x0
                    updated_params = self.mppi_params.copy()
                    updated_params['x0'] = self.current_state.astype(np.float32)
                    self.mppi_controller.set_params(updated_params)
                    self.last_mppi_state = self.current_state.copy()
                elif hasattr(self.mppi_controller, 'params'):
                    self.mppi_controller.params['x0'] = self.current_state.astype(np.float32)
                    self.last_mppi_state = self.current_state.copy()
            
            # Run MPPI solve (no arguments needed)
            optimal_control = self.mppi_controller.solve()
            
            if optimal_control is not None:
                # Get the optimal control sequence
                if hasattr(self.mppi_controller, 'u_cur_d'):
                    # Copy from GPU to CPU
                    u_cur = self.mppi_controller.u_cur_d.copy_to_host()
                    self.current_control = u_cur[0]  # Take first control action
                else:
                    # Fallback: use current control
                    rospy.logwarn("No optimal control returned, using current control")
                
                # Check contact status
                self.contact_active = np.any(np.abs(self.current_state[12:15]) > 0.01)
                
                return True
            else:
                rospy.logwarn("MPPI solve returned None")
                return False
            
        except Exception as e:
            rospy.logerr(f"MPPI optimization failed: {e}")
            return False
    
    def publish_mpc_target(self, target_state):
        """Publish MPC target as PoseStamped"""
        try:
            # Create target message
            target_msg = PoseStamped()
            target_msg.header.stamp = rospy.Time.now()
            target_msg.header.frame_id = "world"
            
            # Set position
            target_msg.pose.position.x = target_state[0]
            target_msg.pose.position.y = target_state[1]
            target_msg.pose.position.z = target_state[2]
            
            # Set orientation (euler to quaternion conversion)
            roll, pitch, yaw = target_state[6], target_state[7], target_state[8]
            
            # Simple euler to quaternion conversion
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            target_msg.pose.orientation.w = cr * cp * cy + sr * sp * sy
            target_msg.pose.orientation.x = sr * cp * cy - cr * sp * sy
            target_msg.pose.orientation.y = cr * sp * cy + sr * cp * sy
            target_msg.pose.orientation.z = cr * cp * sy - sr * sp * cy
            
            # Publish target
            self.mpc_target_pub.publish(target_msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish MPC target: {e}")
    
    def publish_control(self):
        """Publish current control"""
        try:
            # Create control message
            control_msg = Float64MultiArray()
            control_msg.data = self.current_control.tolist()
            
            # Publish control
            self.control_pub.publish(control_msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish control: {e}")
    
    def publish_contact_status(self):
        """Publish contact status"""
        try:
            # Create contact status message
            contact_msg = Bool()
            contact_msg.data = self.contact_active
            
            # Publish contact status
            self.contact_status_pub.publish(contact_msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish contact status: {e}")
    
    def test_loop(self):
        """Test loop that runs independently - OPTIMIZED for speed"""
        # Run at full speed like the original controller
        max_iterations = 100  # Match original test
        
        for i in range(max_iterations):
            if rospy.is_shutdown():
                break
                
            try:
                # Simulate state updates
                self.simulate_state_update()
                
                # Run MPPI optimization
                success = self.run_mppi()
                
                if success:
                    # OPTIMIZATION: Only publish every few iterations to reduce overhead
                    if i % 5 == 0:  # Publish every 5th iteration
                        # Publish current control
                        self.publish_control()
                        
                        # Publish contact status
                        self.publish_contact_status()
                        
                        # Generate and publish MPC target - use the dynamic goal instead of simple offset
                        target_state = self.current_state.copy()
                        # Interpolate between current state and goal for smooth MPC target
                        alpha = 0.3  # How much to move toward goal
                        target_state[0:3] = (1-alpha) * self.current_state[0:3] + alpha * self.goal_state[0:3]
                        self.publish_mpc_target(target_state)
                    
                    # Only log every 10th iteration to reduce overhead
                    if i % 10 == 0:
                        rospy.loginfo(f"Test iteration {self.test_counter}: State={self.current_state[:3]}, Goal={self.goal_state[:3]}, Contact={self.contact_active}")
                else:
                    rospy.logwarn("MPPI optimization failed, skipping control update")
                
            except Exception as e:
                rospy.logerr(f"Error in test loop: {e}")
        
        rospy.loginfo(f"Completed {max_iterations} iterations in test mode")
    
    def update_dynamic_goal(self):
        """Update goal dynamically for more interesting behavior"""
        # Create a moving target that the robot should follow
        time_factor = self.test_counter * 0.05
        
        # Circular motion goal
        goal_x = 0.5 * np.sin(time_factor)
        goal_y = 0.5 * np.cos(time_factor)
        goal_z = -1.0 + 0.2 * np.sin(time_factor * 0.5)  # Varying height
        
        # Update goal state (only position, keep other states at zero)
        self.goal_state[0] = goal_x
        self.goal_state[1] = goal_y
        self.goal_state[2] = goal_z
        
        # Update MPPI goal in controller
        if hasattr(self.mppi_controller, 'set_params'):
            updated_params = self.mppi_params.copy()
            updated_params['xgoal'] = self.goal_state.astype(np.float32)
            self.mppi_params = updated_params  # Update our local copy
            self.mppi_controller.set_params(updated_params)
        elif hasattr(self.mppi_controller, 'params'):
            self.mppi_controller.params['xgoal'] = self.goal_state.astype(np.float32)
    
    def run(self):
        """Main run loop"""
        rospy.loginfo("Starting Fast Hybrid Contact MPPI Test Node")
        
        try:
            # Spin
            rospy.spin()
            
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down Fast Hybrid Contact MPPI Test Node")
        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")

if __name__ == '__main__':
    try:
        node = MPPIHybridContactFastTestNode()
        node.run()
    except Exception as e:
        rospy.logerr(f"Failed to start Fast Hybrid Contact MPPI Test Node: {e}")
        sys.exit(1)
