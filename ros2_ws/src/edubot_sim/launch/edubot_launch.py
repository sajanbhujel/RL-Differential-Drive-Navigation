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

    robot1_description_config = xacro.process_file(
        xacro_file,
        mappings={'robot_ns': 'robot1'}
    )
    robot2_description_config = xacro.process_file(
        xacro_file,
        mappings={'robot_ns': 'robot2'}
    )

    robot1_description = {'robot_description': robot1_description_config.toxml()}
    robot2_description = {'robot_description': robot2_description_config.toxml()}

    world_path = os.path.join(pkg_edubot_sim, 'worlds', 'training_arena.sdf')

    robot1_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='robot1',
        output='screen',
        parameters=[robot1_description, {'use_sim_time': True}]
    )

    robot2_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='robot2',
        output='screen',
        parameters=[robot2_description, {'use_sim_time': True}]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        # launch_arguments={'gz_args': f'-r -s {world_path}'}.items(),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    spawn_robot_1 = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot1/robot_description',
            '-name', 'edubot1',
            '-x', '-2',
            '-y', '2',
            '-z', '0.15'
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
            '-z', '0.15'
        ],
        output='screen'
    )

    delayed_spawn_1 = TimerAction(
        period=3.0,
        actions=[spawn_robot_1]
    )

    delayed_spawn_2 = TimerAction(
        period=4.0,
        actions=[spawn_robot_2]
    )

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

            '/world/training_arena/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen'
    )

    return LaunchDescription([
        robot1_state_publisher,
        robot2_state_publisher,
        gazebo,
        delayed_spawn_1,
        delayed_spawn_2,
        bridge
    ])