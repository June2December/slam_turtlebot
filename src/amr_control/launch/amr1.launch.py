from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    robot_ns = "robot4"

    moveout_node = Node(
        package="amr_control",
        executable="amr1_moveout",
        name="amr_moveout",
        namespace=robot_ns,
        output="screen"
    )

# JH 버전
    # tracking_node = Node(
    #     package="amr_control",
    #     executable="amr1_tracking_aerial",
    #     name="tracking",
    #     namespace=robot_ns,
    #     output="screen",
    #     remappings=[
    #         ('/tf', '/robot4/tf'),
    #         ('/tf_static', '/robot4/tf_static'),
    #     ],
    # )

# CH 버전
    tracking_node = Node(
        package="amr_control",
        executable="amr1_retrack2",
        name="tracking",
        namespace=robot_ns,
        output="screen",
    )



    return LaunchDescription([
        moveout_node,
        tracking_node
    ])