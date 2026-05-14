import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_edubot_sim = get_package_share_directory('edubot_sim')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg_edubot_sim, 'urdf', 'edubot.urdf.xacro')
    world_path = os.path.join(pkg_edubot_sim, 'worlds', 'training_arena.sdf')

    robot_descriptions = {}

    for robot_ns in ['robot1', 'robot2', 'robot3', 'robot4']:
        robot_description_config = xacro.process_file(
            xacro_file,
            mappings={'robot_ns': robot_ns}
        )

        robot_descriptions[robot_ns] = {
            'robot_description': robot_description_config.toxml()
        }

    robot_state_publishers = []

    for robot_ns in ['robot1', 'robot2', 'robot3', 'robot4']:
        robot_state_publishers.append(
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                namespace=robot_ns,
                output='screen',
                parameters=[
                    robot_descriptions[robot_ns],
                    {'use_sim_time': True}
                ]
            )
        )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        # use -s for headless server mode
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
        # launch_arguments={'gz_args': f'-r -s {world_path}'}.items(),
    )

    spawn_robot_1 = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot1/robot_description',
            '-name', 'edubot1',
            '-x', '-2.0',
            '-y', '2.0',
            '-z', '0.15',
            '-Y', '-0.661'
        ],
        output='screen'
    )

    spawn_robot_2 = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot2/robot_description',
            '-name', 'edubot2',
            '-x', '2.5',
            '-y', '2.5',
            '-z', '0.15',
            '-Y', '-2.356'
        ],
        output='screen'
    )

    spawn_robot_3 = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot3/robot_description',
            '-name', 'edubot3',
            '-x', '-1.0',
            '-y', '-3.0',
            '-z', '0.15',
            '-Y', '1.571'
        ],
        output='screen'
    )

    spawn_robot_4 = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot4/robot_description',
            '-name', 'edubot4',
            '-x', '3.0',
            '-y', '0.0',
            '-z', '0.15',
            '-Y', '2.896'
        ],
        output='screen'
    )

    delayed_spawn_1 = TimerAction(period=3.0, actions=[spawn_robot_1])
    delayed_spawn_2 = TimerAction(period=4.0, actions=[spawn_robot_2])
    delayed_spawn_3 = TimerAction(period=5.0, actions=[spawn_robot_3])
    delayed_spawn_4 = TimerAction(period=6.0, actions=[spawn_robot_4])

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/robot1/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/robot1/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/robot1/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',

            '/robot2/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/robot2/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/robot2/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',

            '/robot3/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/robot3/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/robot3/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',

            '/robot4/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/robot4/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/robot4/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        ],
        output='screen'
    )

    return LaunchDescription([
        *robot_state_publishers,
        gazebo,
        delayed_spawn_1,
        delayed_spawn_2,
        delayed_spawn_3,
        delayed_spawn_4,
        bridge,
    ])
