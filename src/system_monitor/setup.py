import os
from setuptools import find_packages, setup

package_name = 'system_monitor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # package dir 수정
    package_data={
        package_name: [
            'UI.html',
            'UI_all_log.html',
            'third_map.png',
        ],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sinya',
    maintainer_email='sinya3443@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'UI_bridge = system_monitor.UI_bridge:main',
            'UI_command = system_monitor.UI_command:main',
            'UI_flask = system_monitor.UI_flask:main',
            'detect = system_monitor.webcam_classifer:main',
            
        ],
    },
)
