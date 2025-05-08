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
            executable='aeb',            
            name='aeb_node',
            output='screen'
        ),
    ])
