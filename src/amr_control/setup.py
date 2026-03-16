from setuptools import find_packages, setup
import glob

package_name = 'amr_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/amr_control/launch', glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sinya',
    maintainer_email='sinya3443@gmail.com',
    description='amr_control_node_package',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'amr1_moveout = amr_control.amr1_moveout_follow_waypoints:main',
            'amr1_track = amr_control.amr1_tracking_aerial_v2:main',
            'amr1_pullout = amr_control.amr1_pullout:main',
        ],
    },
)
