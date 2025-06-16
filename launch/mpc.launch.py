from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = '/home/siddarth/f1ws/src/mercedes/config/mercedes_config.yaml'
    return LaunchDescription([
        Node(
            package='mercedes',
            executable='mpc',
            name='mpc_trajectory_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[config_file],
        ),
        Node(
            package='mercedes',
            executable='dyanamic_trajectory',
            name='dynamic_trajectory_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[{'use_sim_time': True}]
        )
    ])
