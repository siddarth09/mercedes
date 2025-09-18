from setuptools import find_packages, setup
import os 
from glob import glob
from ament_index_python.packages import get_package_share_directory

package_name = 'mercedes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*'))
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='siddarth',
    maintainer_email='siddarth.dayasagar@gmail.com',
    description='Northeastern University - Mercedes-F1/10 Project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aeb = mercedes.ttc:main',
            'teleop = mercedes.teleop_node:main',
            'pid = mercedes.pid_:main', 
            'gui = mercedes.slider_gui:main',
            'waypoint_logger = mercedes.waypoint_logger:main',
            'pure_pursuit = mercedes.pure_pursuit:main',
            'dynamic_trajectory= MMPC.dynamic_trajectory:main',
            'reference_traj_gen=mercedes.reference_traj_gen:main',
            'mpc = MMPC.mpc_tracker:main',
            'mpcc = MMPC.mpcc:main',
            'mpc_traj_tracker = mercedes.mpc_traj_tracker:main',
          
            'reference_traj_gen_sim=mercedes.sim.reference_traj_gen_sim:main',
            'mpc_path_tracker_sim = mercedes.sim.mpc_path_tracker_sim:main',
            'mppi_sim = mercedes.sim.mppi_sim:main',
            'raceline_viz = mercedes.sim.raceline_viz:main',
            'mpcc_sim_raceline = mercedes.sim.mpcc_sim_raceline:main',
            'mpcc_sim_centerline = mercedes.sim.mpcc_sim_centerline:main',

        ],
    },
   
)
