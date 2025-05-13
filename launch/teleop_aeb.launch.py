from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Teleop Keyboard Node
        Node(
            package='mercedes',  
            executable='teleop',  
            name='teleop_ackermann',
            output='screen'
        ),

        # AEB (Automatic Emergency Braking) Node
        Node(
            package='mercedes',  
            executable='waypoint_logger',            
            name='waypoint_logger',
            output='screen'
        ),
    ])
