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

    tracking_node = Node(
        package="amr_control",
        executable="amr1_track",
        name="tracking",
        namespace=robot_ns,
        output="screen",
    )

    pullout_node = Node(
        package="amr_control",
        executable="amr1_pullout",
        name="amr_pullout",
        namespace=robot_ns,
        output="screen"
    )

    return LaunchDescription([
        moveout_node,
        tracking_node,
        pullout_node,
    ])