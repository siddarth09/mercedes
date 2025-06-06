from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = '/home/siddarth/f1ws/src/mercedes/config/mercedes_config.yaml'
    return LaunchDescription([
        Node(
            package='mercedes',
            executable='mpc',
            name='mpc_node',
            output='screen',
            emulate_tty=True,
            parameters=[config_file],
        ),
        # Node(
        #     package='mercedes',
        #     executable='trajectory_generator',
        #     name='trajectory_generator_node',
        #     output='screen',
        #     emulate_tty=True,
        #     parameters=[{'use_sim_time': True}]
        # )
    ])
