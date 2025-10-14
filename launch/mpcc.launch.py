from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Dynamically get the path to the config file in the mercedes package
    mercedes_pkg_path = get_package_share_directory('mercedes')
    config_file = os.path.join(mercedes_pkg_path, 'config', 'mercedes_config.yaml')

    return LaunchDescription([
        Node(
            package='mercedes',
            executable='mpcc',
            name='mpc_trajectory_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[config_file],
        ),

        # Node(
        #     package='mercedes',
        #     executable='aeb',
        #     name='aeb_node',
        #     output='screen',
        #     emulate_tty=True,
        # ),
   ])