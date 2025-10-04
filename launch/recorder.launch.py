from launch import LaunchDescription
from launch.actions import ExecuteProcess
from datetime import datetime

def generate_launch_description():
    # Generate timestamp like 20251003_234500
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"src/mercedes/storage/trainer/record_{timestamp}"

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', output_dir,
                '/scan',
                '/odom',
                '/drive',
                '/tf',
                '/tf_static',
                '/map',
            ],
            output='screen'
        )
    ])
