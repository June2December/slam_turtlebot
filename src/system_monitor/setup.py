from setuptools import find_packages, setup

package_name = 'system_monitor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
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
            'ui_bridge = system_monitor.UI_bridge:main',
            'ui_command = system_monitor.UI_command:main',
            'ui_flask = system_monitor.UI_flask:main',
            'gas_db_logger = system_monitor.v2.gas_db_log:main',
            'gas_dashboard = system_monitor.v1.app:app.run',
            
        ],
    },
)
