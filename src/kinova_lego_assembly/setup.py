"""Setuptools entry point for the bitbots_lego_assembly ROS2 package."""

from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "bitbots_lego_assembly"


def _read_requirements() -> list[str]:
    """Parse requirements.txt, ignoring blank lines and comments."""
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.is_file():
        return []
    return [
        line.strip()
        for line in req_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _model_data_files() -> list[tuple[str, list[str]]]:
    """Install every file under trained_models/ preserving its directory layout."""
    entries: list[tuple[str, list[str]]] = []
    for path in Path("trained_models").rglob("*"):
        if path.is_file():
            install_dir = str(Path("share", package_name, path.parent))
            entries.append((install_dir, [str(path)]))
    return entries


def _config_data_files() -> list[tuple[str, list[str]]]:
    """Install every file under config/ preserving its directory layout."""
    entries: list[tuple[str, list[str]]] = []
    for path in Path("config").rglob("*"):
        if path.is_file():
            install_dir = str(Path("share", package_name, path.parent))
            entries.append((install_dir, [str(path)]))
    return entries


def _asset_data_files() -> list[tuple[str, list[str]]]:
    """Install shared blueprints and mesh assets under the package share.

    Assets live two levels above the ament package root. Colcon requires
    ``data_files`` sources to be relative, so preserve those relative paths.
    """
    asset_root = Path("..") / ".." / "assets"
    entries: list[tuple[str, list[str]]] = []
    blueprint = asset_root / "lego_blueprints.yaml"
    if (Path(__file__).resolve().parent / blueprint).is_file():
        entries.append((str(Path("share", package_name, "assets")), [blueprint.as_posix()]))
    mesh_directory = asset_root / "meshes" / "lego_duplo_bricks"
    meshes = sorted(
        path
        for path in (Path(__file__).resolve().parent / mesh_directory).glob("*.stl")
        if path.is_file()
    )
    if meshes:
        entries.append(
            (
                str(
                    Path(
                        "share",
                        package_name,
                        "assets/meshes/lego_duplo_bricks",
                    )
                ),
                [(mesh_directory / path.name).as_posix() for path in meshes],
            )
        )
    return entries


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (str(Path("share", package_name)), ["package.xml", ".setup_assistant"]),
        (
            str(Path("share", package_name, "launch")),
            glob("launch/*.launch.py"),
        ),
        *_config_data_files(),
        *_asset_data_files(),
        *_model_data_files(),
    ],
    install_requires=[*_read_requirements()],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="LEGO assembly perception and control nodes for the bitbots stack.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            ("lego_segmentation_node = bitbots_lego_assembly.ros.nodes.lego_segmentation:main"),
            ("task_orchestrator_node = bitbots_lego_assembly.ros.nodes.task_orchestrator:main"),
            ("lifecycle_manager_node = bitbots_lego_assembly.ros.nodes.lifecycle_manager:main"),
            (
                "planning_scene_obstacle_publisher_node = "
                "bitbots_lego_assembly.ros.nodes.planning_scene_obstacle_publisher:main"
            ),
        ],
    },
)
