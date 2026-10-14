import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


class FrenetLatticePlanner(Node):
    def __init__(self):
        super().__init__('frenet_lattice_planner')

        self.global_path = []
        self.map_data = None
        self.scan = None
        self.last_scan_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Path, '/global_path', self.path_callback, 10
        )

        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.local_pub = self.create_publisher(
            Path, '/local_path', 10
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        self.create_timer(0.1, self.plan)

        self.get_logger().info('Frenet Lattice Planner started')

    def map_callback(self, msg):
        self.map_data = msg

    def path_callback(self, msg):
        if msg.header.frame_id != 'map':
            return

        self.global_path = [
            (p.pose.position.x, p.pose.position.y)
            for p in msg.poses
        ]

    def scan_callback(self, msg):
        self.scan = msg
        self.last_scan_time = self.get_clock().now()

    def get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time()
            )

            x = tf.transform.translation.x
            y = tf.transform.translation.y

            q = tf.transform.rotation

            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z)
            )

            return x, y, yaw

        except Exception:
            return None

    def world_to_grid(self, x, y):
        if self.map_data is None:
            return None

        info = self.map_data.info

        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        if resolution <= 0:
            return None

        # 지도 원점의 회전각
        q = info.origin.orientation

        yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

        dx = x - origin_x
        dy = y - origin_y

        # 지도 좌표계로 역회전
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

        col = math.floor(local_x / resolution)
        row = math.floor(local_y / resolution)

        if not (0 <= col < info.width and 0 <= row < info.height):
            return None

        return row, col

    def get_map_status(self, x, y):
        if self.map_data is None:
            return "unknown"

        grid = self.world_to_grid(x, y)

        if grid is None:
            return "unknown"

        row, col = grid

        width = self.map_data.info.width
        index = row * width + col

        value = self.map_data.data[index]

        if value < 0:
            return "unknown"

        if value >= 65:
            return "occupied"

        if value <= 25:
            return "free"

        return "unknown"

    def near_existing_wall(self, x, y, tolerance=0.10):
        if self.map_data is None:
            return True

        grid = self.world_to_grid(x, y)

        if grid is None:
            return True

        row, col = grid
        info = self.map_data.info

        radius = math.ceil(tolerance / info.resolution)

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):

                if dr * dr + dc * dc > radius * radius:
                    continue

                nr = row + dr
                nc = col + dc

                if not (0 <= nr < info.height and 0 <= nc < info.width):
                    continue

                index = nr * info.width + nc
                value = self.map_data.data[index]

                if value >= 65:
                    return True

        return False

    def get_obstacles(self):
        if self.scan is None:
            return None

        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                self.scan.header.frame_id,
                Time()
            )
        except Exception:
            return None

        q = tf.transform.rotation

        yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y

        obstacles = []

        for i, r in enumerate(self.scan.ranges):
            if not math.isfinite(r):
                continue

            if r < self.scan.range_min or r > self.scan.range_max:
                continue

            angle = self.scan.angle_min + i * self.scan.angle_increment

            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            gx = tx + lx * math.cos(yaw) - ly * math.sin(yaw)
            gy = ty + lx * math.sin(yaw) + ly * math.cos(yaw)

            obstacles.append((gx, gy))

        return np.array(obstacles).reshape(-1, 2)

    def classify_obstacles(self, obstacles):
        if obstacles is None or self.map_data is None:
            return None

        new_obstacles = []

        for x, y in obstacles:
            status = self.get_map_status(float(x), float(y))

            # 기존 Q1 지도에서 빈 공간이었던 위치만 후보로 분류
            if status != "free":
                continue

            # 기존 벽 근처의 점은 정렬 오차일 가능성이 있음
            if self.near_existing_wall(float(x), float(y)):
                continue

            new_obstacles.append((float(x), float(y)))

        return np.asarray(new_obstacles, dtype=float).reshape(-1, 2)

    def collision_free(self, path, obstacles):
        if obstacles is None:
            return False

        if len(obstacles) == 0:
            return True

        # 차량 반폭 + 안전거리의 보수적 원형 근사
        radius = 0.30

        for px, py in path:
            distances = np.hypot(
                obstacles[:, 0] - px,
                obstacles[:, 1] - py
            )

            if np.any(distances < radius):
                return False

        return True

    def plan(self):
        if not self.global_path or self.scan is None:
            return

        if self.last_scan_time is None:
            return

        age = (
            self.get_clock().now() - self.last_scan_time
        ).nanoseconds / 1e9

        if age > 0.5:
            return

        pose = self.get_pose()

        if pose is None:
            return

        x, y, yaw = pose

        obstacles = self.get_obstacles()

        if obstacles is None:
            return

        path = np.array(self.global_path)

        distances = np.hypot(
            path[:, 0] - x,
            path[:, 1] - y
        )

        nearest = int(np.argmin(distances))

        # 폐곡선 Global Path를 기준으로 전방 약 3m 추출
        reference = []
        total = 0.0
        idx = nearest

        while total < 3.0 and len(reference) < len(path):
            current = path[idx]
            reference.append(current)

            next_idx = (idx + 1) % len(path)
            next_point = path[next_idx]

            total += np.linalg.norm(next_point - current)
            idx = next_idx

        if len(reference) < 3:
            return

        reference = np.array(reference)

        # 횡방향 후보 경로
        offsets = [0.0, 0.15, -0.15, 0.30, -0.30, 0.40, -0.40]

        best_path = None
        best_cost = float('inf')

        n = len(reference)

        for offset in offsets:
            candidate = []

            for i in range(n):
                prev_p = reference[max(i - 1, 0)]
                next_p = reference[min(i + 1, n - 1)]

                dx = next_p[0] - prev_p[0]
                dy = next_p[1] - prev_p[1]

                norm = math.hypot(dx, dy)

                if norm < 1e-6:
                    continue

                nx = -dy / norm
                ny = dx / norm

                # 시작점에서는 원래 경로를 따르고 점진적으로 이동
                progress = i / max(n - 1, 1)

                if progress <= 0.5:
                    t = 2.0 * progress
                    blend = 10*t**3 - 15*t**4 + 6*t**5
                else:
                    t = 2.0 * (progress - 0.5)
                    blend = 1.0 - (10*t**3 - 15*t**4 + 6*t**5)

                d = offset * blend

                candidate.append((
                    reference[i][0] + nx * d,
                    reference[i][1] + ny * d
                ))

            if len(candidate) < 3:
                continue

            if not self.collision_free(candidate, obstacles):
                continue

            cost = abs(offset)

            if cost < best_cost:
                best_cost = cost
                best_path = candidate

        if best_path is None:
            return

        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for px, py in best_path:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(px)
            p.pose.position.y = float(py)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)

        self.local_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrenetLatticePlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
