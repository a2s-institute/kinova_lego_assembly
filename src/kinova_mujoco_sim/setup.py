"""Setup configuration for kinova_mujoco_sim package."""

from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "kinova_mujoco_sim"

# Collect all mesh files (STL) recursively
mesh_files = []
for mesh_path in Path("models").rglob("*.stl"):
    if mesh_path.is_file():
        mesh_files.append(str(mesh_path))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (str(Path("share") / package_name / "launch"), glob("launch/*.launch.py")),
        (str(Path("share") / package_name / "config"), glob("config/*.yaml")),
        (
            str(Path("share") / package_name / "config" / "rviz"),
            glob("config/rviz/*.rviz"),
        ),
        # Robot description xacros (geometry + mujoco_ros2_control interfaces)
        (
            str(Path("share") / package_name / "description"),
            glob("description/*.xacro"),
        ),
        # Arm XML files
        (
            str(Path("share") / package_name / "models/mujoco/arm"),
            glob("models/mujoco/arm/*.xml"),
        ),
        # Arm meshes
        (
            str(Path("share") / package_name / "models/meshes/arm"),
            glob("models/meshes/arm/*.stl"),
        ),
        # Gripper XML files
        (
            str(Path("share") / package_name / "models/mujoco/gripper"),
            glob("models/mujoco/gripper/*.xml"),
        ),
        # Gripper meshes
        (
            str(Path("share") / package_name / "models/meshes/gripper"),
            glob("models/meshes/gripper/*.stl"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Python ROS 2 nodes for Kinova Gen3 + Robotiq 2F-85 MuJoCo simulation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "generate_scene_mjcf = kinova_mujoco_sim.core.generate_scene_mjcf:main",
        ],
    },
)
