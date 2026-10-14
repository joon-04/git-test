import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, DurabilityPolicy

CSV_PATH = '/LIMO_GAZEBO/ws_limo_humble/my_map_global.csv'

class GlobalPathPublisher(Node):
    def __init__(self):
        super().__init__('global_path_publisher')

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.publisher = self.create_publisher(
            Path, '/global_path', qos
        )

        self.data = np.loadtxt(CSV_PATH, delimiter=',', comments='#')
        self.timer = self.create_timer(1.0, self.publish_path)

        self.get_logger().info(
            f'Loaded {len(self.data)} waypoints'
        )

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for row in self.data:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(row[0])
            pose.pose.position.y = float(row[1])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = GlobalPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
