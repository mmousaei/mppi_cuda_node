#!/usr/bin/env python
import rospy
import argparse
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from core_trajectory_msgs.msg import FixedTrajectory
from diagnostic_msgs.msg import KeyValue

class TrajectoryPublisher:
    def __init__(self):
        rospy.init_node('mpc_target_publisher')
        # Hardcoded parameter to choose which publisher to use:
        self.use_pid = True  # Set to True to use the PID fixed trajectory publisher
        
        # Publisher for MPC target trajectory
        self.mpc_pub = rospy.Publisher('/mpc/target', PoseStamped, queue_size=10)
        # Publisher for fixed trajectory (PID debugging)
        self.fixed_traj_pub = rospy.Publisher("/fixed_trajectory", FixedTrajectory, queue_size=10)
        # Get the current odometry once at startup (could also subscribe continuously)
        self.odom = rospy.wait_for_message("/odometry", Odometry)
    
    def send_interpolated_targets(self, start, target, max_step=0.01, pub_rate=20, publisher=None):
        distance = np.linalg.norm(np.array(target) - np.array(start))
        num_steps = int(np.ceil(distance / max_step))
        if publisher is None:
            publisher = self.mpc_pub
        rate = rospy.Rate(pub_rate)
        
        rospy.loginfo("Publishing {} waypoints from {} to {}".format(num_steps, start, target))
        for i in range(1, num_steps + 1):
            alpha = float(i) / num_steps
            pos = np.array(start) + alpha * (np.array(target) - np.array(start))
            
            # Decide which publisher to use based on the hardcoded flag
            if self.use_pid:
                self.publish_position_pid(pos)
            else:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "map"
                pose_msg.pose.position.x = pos[0]
                pose_msg.pose.position.y = pos[1]
                pose_msg.pose.position.z = pos[2]
                # Fixed orientation (identity quaternion)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0

                publisher.publish(pose_msg)
            rospy.loginfo("Published waypoint {}: {}".format(i, pos))
            rate.sleep()
    
    def get_current_position(self):
        pos = self.odom.pose.pose.position
        current = [pos.x, pos.y, pos.z]
        rospy.loginfo("Current position: {}".format(current))
        return current
    
    def publish_trajectory(self, waypoints, max_step=0.01, pub_rate=20):
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            target = waypoints[i + 1]
            self.send_interpolated_targets(start, target, max_step, pub_rate)
    
    def publish_position_pid(self, point):
        """
        Publish a fixed trajectory message for PID debugging.
        """
        x = point[0]
        y = point[1]
        z = point[2]

        traj = FixedTrajectory()
        traj.type = "Point"
        att1 = KeyValue(key="frame_id", value="world")
        att2 = KeyValue(key="height", value=str(z))
        att3 = KeyValue(key="max_acceleration", value=str(0.4))
        att4 = KeyValue(key="velocity", value=str(0.1))
        att5 = KeyValue(key="x", value=str(x))
        att6 = KeyValue(key="y", value=str(y))
        traj.attributes.extend([att1, att2, att3, att4, att5, att6])
        self.fixed_traj_pub.publish(traj)
        rospy.loginfo("Published fixed PID trajectory: {}".format(traj))
    
    def generate_line_trajectory(self, current, target_coords):
        return [current, target_coords]
    
    def generate_circle_trajectory(self, center, plane, radius, num_points=100):
        points = []
        if plane == 'x':
            cx, cy, cz = center
            # Circle in the y-z plane (x remains constant)
            for theta in np.linspace(0, 2 * np.pi, num_points):
                y = cy + radius * np.cos(theta)
                z = cz + radius * np.sin(theta)
                points.append([cx, y, z])
        elif plane == 'y':
            cx, cy, cz = center
            # Circle in the x-z plane (y remains constant)
            for theta in np.linspace(0, 2 * np.pi, num_points):
                x = cx + radius * np.cos(theta)
                z = cz + radius * np.sin(theta)
                points.append([x, cy, z])
        elif plane == 'z':
            cx, cy, cz = center
            # Circle in the x-y plane (z remains constant)
            for theta in np.linspace(0, 2 * np.pi, num_points):
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                points.append([x, y, cz])
        return points
    
    def generate_figure8_trajectory(self, center, plane, amplitude, num_points=100):
        points = []
        if plane == 'x':
            cx, cy, cz = center
            for t in np.linspace(0, 2 * np.pi, num_points):
                y = cy + amplitude * np.sin(t)
                z = cz + amplitude * np.sin(2 * t)
                points.append([cx, y, z])
        elif plane == 'y':
            cx, cy, cz = center
            for t in np.linspace(0, 2 * np.pi, num_points):
                x = cx + amplitude * np.sin(t)
                z = cz + amplitude * np.sin(2 * t)
                points.append([x, cy, z])
        elif plane == 'z':
            cx, cy, cz = center
            for t in np.linspace(0, 2 * np.pi, num_points):
                x = cx + amplitude * np.sin(t)
                y = cy + amplitude * np.sin(2 * t)
                points.append([x, y, cz])
        return points
    
    def run(self, args):
        current_position = self.get_current_position()
        if args.command == 'line':
            target = [args.x, args.y, args.z]
            traj = self.generate_line_trajectory(current_position, target)

            self.publish_trajectory(traj)
        elif args.command == 'circle':
            # Determine a starting point for the circle (angle = 0)
            if args.axis == 'x':
                start_point = [current_position[0],
                                current_position[1] + args.radius,
                                current_position[2]]
            elif args.axis == 'y':
                start_point = [current_position[0] + args.radius,
                                current_position[1],
                                current_position[2]]
            elif args.axis == 'z':
                start_point = [current_position[0] + args.radius,
                                current_position[1],
                                current_position[2]]
            # First, move from current position to the start point
            self.send_interpolated_targets(current_position, start_point)
            circle_points = self.generate_circle_trajectory(current_position, args.axis, args.radius)
            self.publish_trajectory([start_point] + circle_points)
        elif args.command == 'figure8':
            figure8_points = self.generate_figure8_trajectory(current_position, args.axis, args.amplitude)
            self.publish_trajectory([current_position] + figure8_points)
        else:
            rospy.loginfo("No valid trajectory command provided. Use 'line', 'circle', or 'figure8'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Publish different trajectories for MPC target")
    subparsers = parser.add_subparsers(dest='command', help='Trajectory type')
    
    # Line trajectory: e.g., "python mpc_target_publisher.py line 1 2 3"
    parser_line = subparsers.add_parser('line', help='Publish line trajectory')
    parser_line.add_argument('x', type=float, help='Target x coordinate')
    parser_line.add_argument('y', type=float, help='Target y coordinate')
    parser_line.add_argument('z', type=float, help='Target z coordinate')
    
    # Circle trajectory: e.g., "python mpc_target_publisher.py circle x 0.5"
    parser_circle = subparsers.add_parser('circle', help='Publish circle trajectory')
    parser_circle.add_argument('axis', type=str, choices=['x','y','z'], help='Axis that remains constant')
    parser_circle.add_argument('radius', type=float, help='Radius of the circle')
    
    # Figure8 trajectory: e.g., "python mpc_target_publisher.py figure8 x 1"
    parser_figure8 = subparsers.add_parser('figure8', help='Publish figure8 trajectory')
    parser_figure8.add_argument('axis', type=str, choices=['x','y','z'], help='Axis that remains constant')
    parser_figure8.add_argument('amplitude', type=float, help='Amplitude of the figure8')
    
    args = parser.parse_args()
    
    node = TrajectoryPublisher()
    # Change this flag to True to test the PID publisher alternative.
    node.use_pid = False
    node.run(args)
