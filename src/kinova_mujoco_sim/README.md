# kinova_mujoco_sim

MuJoCo simulation of the Kinova Gen3 (7 DoF) work arm and Robotiq 2F-85 gripper
with native `ros2_control`, plus a passive obstacle arm, table, lego holder and
movable bricks. The work arm and gripper are driven through the DFKI
`mujoco_ros2_control/MujocoSystem` plugin, so the standard ROS 2 controllers
(`joint_trajectory_controller`, `GripperActionController`, ...) and tools like
`rqt_joint_trajectory_controller` work out of the box, and `robot_description`
is published for downstream motion planning / MoveIt.

## Layout

| Path | What |
| ---- | ---- |
| `description/` | Top-level work-arm xacro + the `MujocoSystem` `ros2_control` macros. Arm/gripper geometry is reused from the installed `kortex_description` / `robotiq_description` (no fork). |
| `config/` | `ros2_controllers.yaml`, `camera.yaml` (single source of truth for camera topics/frames/intrinsics), `rviz/sim.rviz`. |
| `kinova_mujoco_sim/core/` | Scene builders, layout, transforms, and `generate_scene_mjcf` (dumps the full work cell to a single MJCF). Runs under the plain Python interpreter (no `rclpy`). |
| `kinova_mujoco_sim/ros/` | Launch-time helpers (`launch_helpers.py`): robot description, camera params/remappings, and the generator interpreter/environment. |
| `launch/` | `launch_mujoco_ros2_control.launch.py`. |
| `patches/` | Vendored `mujoco_ros2_control` patch + `apply.sh` (see below). |
| `SCENES.md` | Scene catalog and layout knobs. |

## Prerequisites

- Ubuntu 22.04, ROS 2 Humble.
- Python deps `mujoco==3.9.0` and `pytransform3d` available to the interpreter
  that runs the scene generator (a project venv, or a user-site `pip install`).
  See `requirements.txt`. The launch resolves this interpreter automatically
  (`$KINOVA_MUJOCO_SIM_PYTHON`, an active `$VIRTUAL_ENV`, the workspace `venv/`,
  then system `python3`).
- ROS deps via rosdep (from the workspace root):

  ```bash
  rosdep install --from-paths src/kinova_lego_assembly/src/kinova_mujoco_sim --ignore-src -y
  ```

  `kortex_description` and `robotiq_description` are pulled from the system
  (rosdep) install, not the workspace.

## Build

This package depends on the DFKI `mujoco_ros2_control`, which needs a small set
of local changes (offscreen camera rendering, camera frame/encoding/intrinsics
overrides, PD position-actuator support, continuous ctrl assertion). It is not
released for our use case, so it is vendored: clone it into the workspace at the
pinned base commit and apply the patch. Run the following from the workspace
root (the directory that contains `src/`):

```bash
# 1. Clone the vendored library at the pinned base commit
git clone git@github.com:dfki-ric/mujoco_ros2_control.git src/mujoco_ros2_control
git -C src/mujoco_ros2_control checkout 82b602b

# 2. Apply the vendored patch (idempotent)
src/kinova_lego_assembly/src/kinova_mujoco_sim/patches/apply.sh

# 3. Resolve ROS dependencies for everything in the workspace. The patch from
#    step 2 marks mujoco_ros2_control_examples as ignored, so its extra deps
#    (ur_description, franka_description, ...) are not pulled in here.
rosdep install --from-paths src --ignore-src -y

# 4. Install the Python requirements into the interpreter that will run the build
#    and the scene generator (activate your venv first, or use a user-site install).
python3 -m pip install -r src/kinova_lego_assembly/src/kinova_mujoco_sim/requirements.txt

# 5. Build kinova_mujoco_sim and its dependencies (incl. mujoco_ros2_control).
#    Invoke colcon via `python3 -m colcon` so it runs under the same interpreter
#    that has the requirements above, ensuring the Python deps are found.
python3 -m colcon build --packages-up-to kinova_mujoco_sim --symlink-install
```

> If the released `ros-humble-mujoco-ros2-control` Debian package is installed,
> the workspace build of the patched source overrides it and colcon will refuse
> the override. Remove the released package once so the workspace copy is the
> only one (`sudo apt remove ros-humble-mujoco-ros2-control`); otherwise add
> `--allow-overriding mujoco_ros2_control` to the build.

See [`patches/README.md`](patches/README.md) for what the patch changes and how
to regenerate it.

## Run

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch kinova_mujoco_sim launch_mujoco_ros2_control.launch.py
```

**Stopping:** press `Ctrl+C` in the launch terminal. This shuts down every
launched node and works the same in headless mode. Closing the MuJoCo GUI
window does not stop the nodes.

Drive the arm with, e.g.:

```bash
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
```

## Cameras

`config/camera.yaml` is the single source of truth for the wrist camera topics,
frame ids, encodings, resolutions and intrinsics. The launch maps the camera
node's fixed leaf topics onto the `topic` / `info_topic` names from the yaml via
remappings, so editing those values changes the published topic names without a
library change. Frame ids, encoding (e.g. `16UC1` depth) and intrinsics are
applied by the vendored camera patch.

The colour-registered point cloud is published from the RGB camera instance
(`point_cloud: true` on `wrist`), not from the native-depth instance. That
matches the real RealSense contract used by MMOR: `/camera/depth/color/points`
is an organised 640x480 cloud in `camera_color_frame`, with invalid depth as
NaN. Raw depth stays at 480x270 in `camera_depth_frame`.

## Spawn pose and continuous joints

`config/home_pose.yaml` is the MuJoCo / ros2_control **spawn** configuration,
not the orchestrator home viewpoint. The assembly tree's first move is a
joint-space goal to `orchestrator.home_joint_positions` (OMPL RRTConnect), so
the spawn pose must be collision-free with a collision-free joint path to home
(continuous joints 1/3/5/7 are bounded to ±π in the URDF by
`bound_continuous_joints`). The spawn in
`home_pose.yaml` is home with a small joint offset so the arm visibly moves
to the viewpoint on start.

`joint_trajectory_controller` goal tolerances in `config/ros2_controllers.yaml`
must stay enabled. With them at 0.0 the controller reports success even when
MuJoCo clamps a command.

## Development notes

### Tool transform (arm flange to gripper base)

The gripper is mounted at the arm's `end_effector_link`; the offset comes from
the end-effector frame in `models/mujoco/arm/gen3.xml` (matched by the kortex
`end_effector` joint in the URDF). The gripper attachment is built in
`kinova_mujoco_sim/core/scene/stage.py`.

| Parameter     | Value        | Source                                          |
| ------------- | ------------ | ----------------------------------------------- |
| Translation X | 0.000 m      | attach frame `pos = [0, 0, 0]`                  |
| Translation Y | 0.000 m      | attach frame `pos = [0, 0, 0]`                  |
| Translation Z | -0.0615 m    | end-effector frame `pos="0 0 -0.061525"`        |
| Rotation      | 180 deg about X | end-effector frame `quat="0 1 0 0"`          |

See [ADR-0012](../../docs/adr/0012-kinova-mujoco-sim-design.md) for the rationale
of the MuJoCo control ranges for the joints.

## Documentation

- [`SCENES.md`](SCENES.md) - scene catalog and layout knobs.
- [`patches/README.md`](patches/README.md) - vendored `mujoco_ros2_control` changes.
- `docs/adr/0015-*` (ros2_control bringup, supersedes `0012-*`), `docs/adr/0013-*` (scene layout) (repo root) - design rationale.
