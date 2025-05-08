import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import sys
import select
import termios
import tty
import numpy as np

class TeleopAckermann(Node):
    def __init__(self):
        super().__init__('teleop_ackermann')
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Control variables
        self.speed = 0.0
        self.steering_angle = 0.0
        self.speed_step = 0.1
        self.steering_step = 0.1
        self.max_speed = 3.0
        self.max_steering = 0.4189

        # Terminal settings
        self.settings = termios.tcgetattr(sys.stdin)

        self.get_logger().info("Use W/S to increase/decrease speed, A/D to steer, S to stop. Ctrl+C to exit cleanly.")

        self.timer = self.create_timer(0.1, self.send_cmd)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = ''
        if rlist:
            key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def send_cmd(self):
        try:
            key = self.get_key()
        except Exception as e:
            self.get_logger().error(f'Exception while reading key: {e}')
            return
        
        

        if key == 'w':
            self.speed=min(self.speed + self.speed_step, self.max_speed)
            self.speed = np.clip(self.speed + self.speed_step, -self.max_speed, self.max_speed)
        elif key == 'x':
            self.speed = max(self.speed - self.speed_step, -self.max_speed)
            self.speed = np.clip(self.speed - self.speed_step, -self.max_speed, self.max_speed)
        elif key == 'a':
            self.steering_angle = np.clip(self.steering_angle + self.steering_step, -self.max_steering, self.max_steering)
        elif key == 'd':
            self.steering_angle = np.clip(self.steering_angle - self.steering_step, -self.max_steering, self.max_steering)
        elif key == 's':
            self.speed = 0.0
            self.steering_angle = 0.0

        cmd = AckermannDriveStamped()
        cmd.drive.speed = self.speed
        cmd.drive.steering_angle = self.steering_angle

        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        """Override destroy_node to reset terminal settings."""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TeleopAckermann()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
