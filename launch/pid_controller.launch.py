from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('mercedes'),
        'config',
        'mercedes_config.yaml'
    )

    return LaunchDescription([
        Node(
            package='mercedes',
            executable='pid',
            name='pid_controller',
            output='screen',
            parameters=[config_path]
        ),

        # Node(
        #     package='mercedes',  
        #     executable='aeb',            
        #     name='aeb_node',
        #     output='screen'
        # ),
        
        
    ])
