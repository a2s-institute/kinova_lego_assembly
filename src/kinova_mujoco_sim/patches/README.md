# mujoco_ros2_control patch

`mujoco_ros2_control.patch` collects the changes `kinova_mujoco_sim` needs in the
vendored DFKI `mujoco_ros2_control` package. They live in the simulator's
renderer / process main loop / hardware mapping and therefore cannot be made from
`kinova_mujoco_sim` alone. Everything else (models, xacros, launch, controller and
camera configs, scene generation) stays in `kinova_mujoco_sim`.

- Base commit: `82b602bb8d46209aa2479c2cc7c6ecf5361f908c`
  (`adapt migrated examples and plugin to humble`)
- Target repo path: `src/mujoco_ros2_control` (the patch touches the
  `mujoco_ros2_control` lib plus ignore markers in `mujoco_ros2_control_examples`)

## What it changes and why

1. Offscreen camera rendering (`src/mujoco_depth_camera.cpp`)
   The camera rendered into a hidden GLFW window framebuffer and read it back,
   which captured the host screen on this GL/compositor stack (the published image
   was the desktop, at the default 640x480). Switched to MuJoCo's offscreen
   framebuffer (`mjFB_OFFSCREEN`) sized by the model `visual/global` offwidth/
   offheight, and render at the configured resolution.

2. Camera frame/encoding/intrinsics overrides (`src/mujoco_depth_camera.cpp`,
   `src/mujoco_rgbd_camera_parameters.yaml`)
   Added `color_frame_id`, `depth_frame_id`, `color_encoding` (`bgr8` or
   `rgb8`), `depth_in_millimeters`, `camera_info_k`, `camera_info_p`,
   `camera_info_r` so the published streams match a fixed
   frame/encoding/intrinsics contract. Depth is optionally published as 16UC1
   millimeters (else 32FC1 metres); explicit k/p override the FOV-derived
   intrinsics; explicit r overrides the default identity rectification.
   Topic names are not part of this patch:
   `kinova_mujoco_sim.ros.launch_helpers.camera_node_params_file` reads
   `config/camera.yaml` and emits launch-level remappings applied to the
   `mujoco_ros2_control` node, mapping the node's fixed leaf topics
   (`color/image_raw`, `color/camera_info`, `depth/image_rect_raw`,
   `depth/camera_info`, `depth/points`) onto the
   `topic` / `info_topic` / `points_topic` from `config/camera.yaml`.

3. PD position actuators (`src/mujoco_system.cpp`)
   The position-actuator classifier required `biasprm[2]==0` (kv==0), so the
   Kinova MuJoCo Menagerie `<position kp kv>` PD actuators were not recognised and
   the joint trajectory controller's position commands were silently dropped. The
   classifier now allows a non-zero `kv` term (velocity actuators still never match
   because they have `biasprm[1]==0`).

4. Continuous ctrl assertion (`src/mujoco_system.cpp`)
   `write()` only set `mjData.ctrl` when the command value changed (`position !=
   last_command`), relying on the value persisting in `mjData`. Because the GUI's
   `mjData` shares the same `ctrl` array (the plugin aliases it), moving a control
   slider in the simulate GUI overwrites the actuator while `last_command` stays
   stale; re-sending the same goal value was then a no-op and the joint (e.g. the
   gripper) never returned to the controller's command. `write()` now writes `ctrl`
   every cycle for managed POSITION/VELOCITY actuators, so ros2_control stays
   authoritative and GUI edits to managed actuators are reasserted on the next step.

5. Add two empty marker files into `mujoco_ros2_control_examples` to make both colcon
   and rosdep ignore the examples package, so those deps are never required.

6. Unbounded joint lower limit (`include/mujoco_ros2_control/mujoco_system.hpp`)
   `JointData::lower_limit` defaulted to `std::numeric_limits<double>::min()`,
   which is the smallest *positive* normalised double. Every negative position
   command on a joint without URDF limits (the Gen3 continuous joints) was
   therefore clamped to ~0. The default is now `lowest()`.

7. Calibrated RGB-D contract (`src/mujoco_depth_camera.cpp`)
   When `camera_info_k` is set, the GL frustum uses those fx/fy/cx/cy values
   (including an off-centre principal point) and the render aspect is chosen
   so both focal lengths survive the resize to the published resolution.
   Invalid depth (pixels at the far plane) is published as NaN in the organised
   cloud, matching a real depth camera rather than a finite point at the
   camera origin.

## Apply

Use the helper script (idempotent; resolves the target repo relative to the
workspace, or pass it explicitly):

```bash
src/bitbots_lego_assembly/src/kinova_mujoco_sim/patches/apply.sh
# or: apply.sh /path/to/src/mujoco_ros2_control
```

Equivalent manual apply:

```bash
cd src/mujoco_ros2_control
git apply ../bitbots_lego_assembly/src/kinova_mujoco_sim/patches/mujoco_ros2_control.patch
```

(Adjust the relative path to wherever `kinova_mujoco_sim` is checked out.)

## Rebuild

`mujoco_ros2_control` is a normal workspace dependency of `kinova_mujoco_sim`,
so a `--packages-up-to` build picks it up:

```bash
colcon build --packages-up-to kinova_mujoco_sim --symlink-install
```

If the released `ros-humble-mujoco-ros2-control` Debian package is installed,
either remove it once (`sudo apt remove ros-humble-mujoco-ros2-control`) so the
workspace copy is authoritative, or append `--allow-overriding mujoco_ros2_control`.

## Regenerate this patch

After further edits in the lib. The `git add -N` marks any new files (e.g. the
examples ignore markers) as intent-to-add so `git diff` includes them:

```bash
cd src/mujoco_ros2_control
git add -N mujoco_ros2_control_examples/COLCON_IGNORE mujoco_ros2_control_examples/AMENT_IGNORE
git diff > ../bitbots_lego_assembly/src/kinova_mujoco_sim/patches/mujoco_ros2_control.patch
git reset -q
```
