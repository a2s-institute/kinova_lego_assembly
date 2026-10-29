# kinova_mujoco_sim

MuJoCo simulation of the Kinova Gen3 (7 DoF) work arm and Robotiq 2F-85 gripper
with native `ros2_control`, plus a passive obstacle arm, table, lego holder and
movable bricks. The work arm and gripper are driven through the DFKI
`mujoco_ros2_control/MujocoSystem` plugin, so the standard ROS 2 controllers
(`joint_trajectory_controller`, `GripperActionController`, ...) and tools like
`rqt_joint_trajectory_controller` work out of the box, and `robot_description`
is published for downstream motion planning / MoveIt.

## Installation

### Pre-requisites

> **Install the main application first.** The simulator runs alongside
> `kinova_lego_assembly` in the same workspace and venv. If you have not done
> so yet, complete steps 1 to 4 of the root README's
> [Getting started](../../README.md#getting-started) before continuing here.

Once the build below finishes, run the full pipeline in simulation as described
in [Run with the assembly pipeline](#run-with-the-assembly-pipeline).

### Build

This package depends on the DFKI `mujoco_ros2_control`, which needs a small set
of local changes (offscreen camera rendering, camera frame/encoding/intrinsics
overrides, PD position-actuator support, continuous ctrl assertion). It is not
released for our use case, so it is vendored: clone it into the workspace at the
pinned base commit and apply the patch.:

```bash
cd ~/kinova_lego_assembly_ws

# 1. Clone the vendored library at the pinned base commit
git clone git@github.com:dfki-ric/mujoco_ros2_control.git src/mujoco_ros2_control
git -C src/mujoco_ros2_control checkout 82b602b

# 2. Apply the vendored patch
src/kinova_lego_assembly/src/kinova_mujoco_sim/patches/apply.sh

# 3. Resolve ROS dependencies for everything in the workspace.
rosdep install --from-paths src --ignore-src -y

# 4. Activate venv and install the Python requirements
source venv/bin/activate
python3 -m pip install -r src/kinova_lego_assembly/src/kinova_mujoco_sim/requirements.txt

# 5. Build kinova_mujoco_sim and its dependencies
python3 -m colcon build --packages-up-to kinova_mujoco_sim --symlink-install
```

> If the released `ros-humble-mujoco-ros2-control` Debian package is installed,
> the workspace build of the patched source overrides it and colcon will refuse
> the override. Remove the released package once so the workspace copy is the
> only one (`sudo apt remove ros-humble-mujoco-ros2-control`); otherwise add
> `--allow-overriding mujoco_ros2_control` to the build.

See [`patches/README.md`](patches/README.md) for what the patch changes and how
to regenerate it.

### Run

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
source venv/bin/activate
ros2 launch kinova_mujoco_sim launch_mujoco_ros2_control.launch.py
```

**Stopping:** press `Ctrl+C` in the launch terminal. This shuts down every
launched node and works the same in headless mode. Closing the MuJoCo GUI
window does not stop the nodes.

Drive the arm with, e.g.:

```bash
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
```

### Run with MoveIt

`launch_sim_moveit.launch.py` starts the simulation, then `move_group` and a
MoveIt RViz view. The MoveIt config for the simulated arm (`config/gen3.srdf`,
kinematics, joint limits, planners, controllers) lives in this package.

```bash
ros2 launch kinova_mujoco_sim launch_sim_moveit.launch.py
```

### Run with the assembly pipeline

```bash
ros2 launch kinova_lego_assembly launch_task_orchestrator.launch.py sim_mode:=true
```

## Configuration

### Cameras

`config/camera.yaml` is the single source of truth for the wrist camera topics,
frame ids, encodings, resolutions and intrinsics. Frame ids, encoding
(e.g. `16UC1` depth) and intrinsics are applied by the vendored camera patch.

The colour-registered point cloud is published from the RGB camera instance
(`point_cloud: true` on `wrist`), not from the native-depth instance. 
`/camera/depth/color/points` is an organised 640x480 cloud in `camera_color_frame`,
with invalid depth as NaN. Raw depth stays at 480x270 in `camera_depth_frame`.

### Spawn pose and continuous joints

`config/home_pose.yaml` is the MuJoCo / ros2_control **spawn** configuration,
not the orchestrator home viewpoint.

`joint_trajectory_controller` goal tolerances in `config/ros2_controllers.yaml`
must stay enabled. With them at 0.0 the controller reports success even when
MuJoCo clamps a command.

## Development notes

### Tool transform (arm flange to gripper base)

The gripper is mounted at the arm's `end_effector_link`; the offset comes from
the end-effector frame in `models/mujoco/arm/gen3.xml` (matched by the kortex
`end_effector` joint in the URDF). The gripper attachment is built in
`kinova_mujoco_sim/core/scene/stage.py`.

| Parameter     | Value           | Source                                   |
| ------------- | --------------- | ---------------------------------------- |
| Translation X | 0.000 m         | attach frame `pos = [0, 0, 0]`           |
| Translation Y | 0.000 m         | attach frame `pos = [0, 0, 0]`           |
| Translation Z | -0.0615 m       | end-effector frame `pos="0 0 -0.061525"` |
| Rotation      | 180 deg about X | end-effector frame `quat="0 1 0 0"`      |

See [ADR-0012](../../docs/adr/0012-kinova-mujoco-sim-design.md) for the rationale
of the MuJoCo control ranges for the joints.

## Documentation

- [`SCENES.md`](SCENES.md) - scene catalog and layout knobs.
- [`patches/README.md`](patches/README.md) - vendored `mujoco_ros2_control` changes.
- `docs/adr/0015-*` (ros2_control bringup, supersedes `0012-*`), `docs/adr/0013-*` (scene layout) (repo root) - design rationale.
