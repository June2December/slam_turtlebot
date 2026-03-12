from setuptools import find_packages, setup

package_name = 'detection'

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
            'capture_node = detection.capture_turtle4:main',
            'rgb_detect = detection.obj_det_dual:main',
            'amr2_detect_old = detection.obj_det_depth:main',
            'amr2_detect = detection.obj_det_amr2:main',
            'amr2_detect_unkonwn = detection.obj_det_depth_unknown:main',
            'amr1_detect_test1 = detection.depth_aligned_dual:main',
            'amr1_detect_test2 = detection.amr1_observe_v2_detect_loop:main'
        ],
    },
)
