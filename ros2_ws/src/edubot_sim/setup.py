from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'edubot_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sajan',
    maintainer_email='your_email@example.com',
    description='EduBot simulation package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_agent = edubot_sim.train_agent:main',
            'save_lidar = edubot_sim.save_lidar:main',
            'reset_robot = edubot_sim.reset_robot:main',
        ],
    },
)