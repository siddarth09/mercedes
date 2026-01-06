from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Package paths
    mercedes_dir = get_package_share_directory('mercedes')

    # Map files
    loc_map_yaml  = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/maps/csc433_local.yaml'        # for AMCL
    plan_map_yaml = '/home/deepak/Data/f1tenth/mercedes_ws/src/mercedes/maps/csc433_track.yaml'        # for planning

    # AMCL parameter file
    amcl_params = PathJoinSubstitution([
        FindPackageShare('mercedes'),
        'config',
        'amcl_config.yaml'
    ])

    return LaunchDescription([
        # Map Server (Localization)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server_localization',
            output='screen',
            parameters=[{
                'yaml_filename': loc_map_yaml,
                'frame_id': 'map',
                'use_sim_time': True,
            }],
            # Publish on a dedicated topic
            remappings=[('map', 'loc_map')],
        ),

        # Map Server (Planning)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server_planning',
            output='screen',
            parameters=[{
                'yaml_filename': plan_map_yaml,
                'frame_id': 'map',          # keep the same frame; different topic isolates it
                'use_sim_time': True,
            }],
            remappings=[('map', 'plan_map')],
        ),

        # AMCL node (subscribe to localization map)
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[amcl_params],
            # Force AMCL to read the localization map
            remappings=[('map', 'loc_map')],
        ),

        # Lifecycle Manager (bring up both map servers + AMCL)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': [
                    'map_server_localization',
                    'map_server_planning',
                    'amcl'
                ],
            }],
        ),
    ])
