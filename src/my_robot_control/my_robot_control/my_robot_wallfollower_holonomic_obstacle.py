#!/usr/bin/env python3
"""Holonomic obstacle wall follower implementation."""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


def clamp(value, low, high):
    return max(low, min(high, value))


def euler_from_quaternion_yaw(x, y, z, w):
    """Calcula la rotación yaw (en radianes) a partir de un cuaternión."""
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class HolonomicObstacleWallFollower(Node):
    """Holonomic controller with state machine for obstacle avoidance."""

    def __init__(self):
        super().__init__('holonomic_obstacle_wall_follower')

        self.follow_distance = float(
            self.declare_parameter('follow_distance', 0.35).value
        )
        self.max_front_length = float(
            self.declare_parameter('max_front_length', 0.50).value
        )
        self.forward_speed = float(
            self.declare_parameter('forward_speed', 0.20).value
        )
        self.lateral_speed = float(
            self.declare_parameter('lateral_speed', 0.20).value
        )
        self.turn_speed = float(
            self.declare_parameter('turn_speed', 1.00).value
        )
        self.detection_limit = float(
            self.declare_parameter('detection_limit', 1.20).value
        )
        self.lateral_gain = float(
            self.declare_parameter('lateral_gain', 2.0).value
        )
        self.alignment_gain = float(
            self.declare_parameter('alignment_gain', 2.0).value
        )

        self.RIGHT = 'RIGHT'
        self.FRONT = 'FRONT'
        self.FRONT_RIGHT = 'FRONT_RIGHT'
        self.BACK_RIGHT = 'BACK_RIGHT'

        self.diagonal_factor = 0.5
        self.region_confirmations = 3
        self.robust_rays = 3

        self.state = self.RIGHT
        self.obstacle_is_long = False
        self.passing_short_obstacle = False
        self.command = Twist()
        self.odom_pose = None
        self.front_start_pose = None
        self.right_angle = None

        self.region_candidate = None
        self.region_candidate_count = 0
        self.stable_region = None

        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.publish_command)

        self.get_logger().info('Nodo seguidor de pared holonomico iniciado.')

    def publish_command(self):
        self.publisher.publish(self.command)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = euler_from_quaternion_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.odom_pose = (position.x, position.y, yaw)

    def region_measurements(self, scan):
        regions = {
            self.FRONT: [],
            self.FRONT_RIGHT: [],
            self.RIGHT: [],
            self.BACK_RIGHT: []
        }

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                continue

            angle_rad = angle_min + i * angle_increment
            angle_deg = math.degrees(angle_rad)
            angle_deg = (angle_deg + 180) % 360 - 180

            if -20.0 <= angle_deg <= 20.0:
                regions[self.FRONT].append((r, angle_deg))
            elif -70.0 <= angle_deg < -20.0:
                regions[self.FRONT_RIGHT].append((r, angle_deg))
            elif -110.0 <= angle_deg < -70.0:
                regions[self.RIGHT].append((r, angle_deg))
            elif -160.0 <= angle_deg < -110.0:
                regions[self.BACK_RIGHT].append((r, angle_deg))

        measurements = {}
        for region_name, rays in regions.items():
            if not rays:
                measurements[region_name] = (math.inf, None)
            else:
                rays.sort(key=lambda x: x[0])
                nearest = rays[:self.robust_rays]
                mean_dist = sum(ray[0] for ray in nearest) / len(nearest)
                mean_angle = sum(ray[1] for ray in nearest) / len(nearest)
                measurements[region_name] = (mean_dist, mean_angle)

        return measurements

    def closest_region(self, measurements):
        valid_regions = {
            reg: dist_angle[0] 
            for reg, dist_angle in measurements.items() 
            if dist_angle[0] < self.detection_limit
        }
        if not valid_regions:
            return None
        return min(valid_regions, key=valid_regions.get)

    def confirmed_region(self, measurements):
        candidate = self.closest_region(measurements)

        if candidate == self.region_candidate:
            self.region_candidate_count += 1
        else:
            self.region_candidate = candidate
            self.region_candidate_count = 1

        if self.region_candidate_count >= self.region_confirmations:
            self.stable_region = self.region_candidate

        return self.stable_region

    def lateral_progress(self):
        if self.odom_pose is None or self.front_start_pose is None:
            return 0.0

        x, y, _ = self.odom_pose
        x0, y0, yaw0 = self.front_start_pose

        dx = x - x0
        dy = y - y0

        progress = abs(-math.sin(yaw0) * dx + math.cos(yaw0) * dy)
        return progress

    def follow_right(self, measurements, closest, cmd):
        if closest == self.FRONT:
            self.state = self.FRONT
            self.front_start_pose = self.odom_pose
            self.obstacle_is_long = False
            return

        dist, angle = measurements[self.RIGHT]
        if not math.isinf(dist):
            cmd.linear.x = self.forward_speed

            error_dist = dist - self.follow_distance
            cmd.linear.y = clamp(self.lateral_gain * error_dist, -self.lateral_speed, self.lateral_speed)

            if not self.passing_short_obstacle and angle is not None:
                error_angle = math.radians(angle - (-90.0))
                cmd.angular.z = clamp(self.alignment_gain * error_angle, -self.turn_speed, self.turn_speed)
            else:
                cmd.angular.z = 0.0

    def follow_front(self, measurements, closest, cmd):
        progress = self.lateral_progress()

        if progress >= self.max_front_length:
            self.obstacle_is_long = True
            cmd.angular.z = self.turn_speed
            if closest == self.RIGHT or closest == self.FRONT_RIGHT:
                self.state = self.RIGHT
        else:
            cmd.linear.x = 0.0
            cmd.linear.y = self.lateral_speed
            
            if closest != self.FRONT:
                self.passing_short_obstacle = True
                self.state = self.FRONT_RIGHT

    def follow_front_right(self, measurements, closest, cmd):
        if self.obstacle_is_long:
            cmd.angular.z = self.turn_speed
            if closest == self.RIGHT:
                self.state = self.RIGHT
        else:
            cmd.linear.x = self.forward_speed * self.diagonal_factor
            cmd.linear.y = self.lateral_speed * self.diagonal_factor

            if closest == self.RIGHT:
                self.state = self.RIGHT
            elif closest == self.BACK_RIGHT:
                self.state = self.BACK_RIGHT

    def follow_back_right(self, measurements, closest, cmd):
        cmd.linear.x = self.forward_speed
        cmd.linear.y = -self.lateral_speed * self.diagonal_factor

        if closest == self.RIGHT:
            self.passing_short_obstacle = False
            self.state = self.RIGHT

    def scan_callback(self, scan):
        measurements = self.region_measurements(scan)
        closest = self.confirmed_region(measurements)

        cmd = Twist()

        if self.state == self.RIGHT:
            self.follow_right(measurements, closest, cmd)
        elif self.state == self.FRONT:
            self.follow_front(measurements, closest, cmd)
        elif self.state == self.FRONT_RIGHT:
            self.follow_front_right(measurements, closest, cmd)
        elif self.state == self.BACK_RIGHT:
            self.follow_back_right(measurements, closest, cmd)

        self.command = cmd

    def stop(self):
        self.command = Twist()
        self.timer.cancel()
        try:
            self.publisher.publish(self.command)
        except Exception:
            pass


def main(args=None):
    rclpy.init(
        args=args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = HolonomicObstacleWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
