# kinova_lego_assembly

ROS 2 (Humble) package providing LEGO perception and assembly nodes. Its
segmentation service wraps a pure-Python
[Ultralytics YOLO](https://docs.ultralytics.com/) backend exposed through
`~/segment_image`; its task orchestrator coordinates perception and motion
planning to assemble a LEGO model.
The service uses
[`SegmentImage.srv`](../kinova_lego_assembly_interfaces/srv/SegmentImage.srv).

The implementation keeps a clear split between ROS and core logic:

- [`kinova_lego_assembly/core/`](kinova_lego_assembly/core) holds the
  ROS-free Python (segmentation backends, dataclasses, model registry).
- [`kinova_lego_assembly/ros/`](kinova_lego_assembly/ros) holds the ROS 2
  nodes, including segmentation and task orchestration.

## Prerequisites

- ROS2 Humble (or compatible) sourced in your shell.
- Python 3.10 (recommended to use virtual env via
  [`uv`](https://docs.astral.sh/uv/) that has access to system site packages).
- [`rosdep`](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html)
  initialised on the machine.

## Shared dependencies

Activate your virtual environment before installing Python dependencies.
Run the pip and `rosdep` commands below from the workspace root; run pytest
from the package repository root (`src/kinova_lego_assembly` inside the
workspace). `colcon` and `ros2` commands run from the workspace root.

Install the package's Python runtime dependencies from the workspace root:

```bash
# from the workspace root
python3 -m pip install -r src/kinova_lego_assembly/requirements.txt
# optional: install test and lint tools
python3 -m pip install -r src/kinova_lego_assembly/requirements.dev.txt
```

The root README's workspace setup clones `mas_industrial_robotics`, which
provides the application's `mir_interfaces` dependency and the orchestrator's
MMOR object-recognition and pose-estimation packages. Resolve ROS dependencies
from the workspace root:

```bash
# from the workspace root
rosdep install \
  --from-paths src/kinova_lego_assembly/src src/mas_industrial_robotics \
  --ignore-src -y
```

ROS dependencies are declared for the package as a whole, so `rosdep` installs
dependencies used by both launch paths, including MMOR.

## Download trained models

Model weights are not tracked in git. Download the latest bundle from
[this OneDrive folder](https://studhbrsde-my.sharepoint.com/:f:/g/personal/scotrraaj_gopal_smail_inf_h-brs_de/IgBjYcjkEeCuRKjxh-8T2YHeAdhgZvkcBIdpvcioWX8hLjs?e=ai2MSA)
and extract it into
[`trained_models/`](trained_models) so the layout matches
[`trained_models/README.md`](trained_models/README.md):

The registry picks the highest available `v<N>` directory by default;
override with the `segmentation.model_version` parameter.

## Build

Build the interface, MIR interface, and application packages for segmentation
(from the workspace root):

```bash
python3 -m colcon build --packages-up-to \
  kinova_lego_assembly_interfaces mir_interfaces kinova_lego_assembly
source install/setup.bash
```

## Run

### Segmentation

The segmentation launch starts the `LegoSegmentation` service node, which
provides `~/segment_image`. Run it from the workspace root with the workspace
sourced:

```bash
ros2 launch kinova_lego_assembly launch_kinova_lego_assembly.launch.py
```

Tune segmentation inference parameters through ROS parameters (see
[`config/params.yml`](config/params.yml)), for example:

```bash
ros2 param set /LegoSegmentation segmentation.yolo.conf 0.7
```

### Orchestrator

`launch_task_orchestrator.launch.py` starts the assembly pipeline in order:
the segmentation service, MMOR object recognition and its lifecycle manager,
the planning-scene obstacle publisher, and the task orchestrator. It also
starts the task RViz view and waits for perception and motion-planning
dependencies before executing the assembly behavior.

MMOR's object recognition estimates object poses from camera data. Its
`mir_object_recognition` and `mir_interfaces` packages come from
[`mas_industrial_robotics`](https://github.com/Jay-droid/mas_industrial_robotics),
which you cloned under **Shared dependencies** above. The
orchestrator needs this repository even when using a real robot.

The launch defaults to `sim_mode:=false`, which assumes a real robot and an
external robot/MoveIt stack are already running. The launch imports
`kinova_mujoco_sim` even in real-robot mode, so first prepare its dependencies
by following the **Build** instructions in the
[MuJoCo simulator README](../kinova_mujoco_sim/README.md). Then build the
orchestrator and its in-workspace dependencies (from the workspace root):

```bash
colcon build --packages-up-to kinova_lego_assembly
source install/setup.bash
```

To launch the MuJoCo simulator and MoveIt together with the assembly pipeline:

```bash
ros2 launch kinova_lego_assembly launch_task_orchestrator.launch.py \
  sim_mode:=true use_sim_time:=true
```

For real-robot operation, run the same launch with the default
`sim_mode:=false`, while the robot and MoveIt stack are running externally:

```bash
ros2 launch kinova_lego_assembly launch_task_orchestrator.launch.py
```

## Tests

Tests are split between pure-Python (`test/core/`) and ROS-dependent
(`test/ros/`) suites, with shared assets under `test/assets/`.

List everything [pytest](https://docs.pytest.org/) would collect (no execution):

```bash
# from src/kinova_lego_assembly
pytest --collect-only
```

Run the YOLO segmentor integration tests only:

```bash
# from src/kinova_lego_assembly
pytest src/kinova_lego_assembly/test/core/test_yolo_segmentor.py
```

Run the offline motion/grasp scoring tests with per-pose pass-rate output:

```bash
# from the repo root (kinova_lego_assembly/)
PYTHONPATH=$PWD/src/kinova_lego_assembly pytest -v -s --log-cli-level=INFO \
  src/kinova_lego_assembly/test/motion/test_grasp_offline_scoring.py
```

These tests are ROS-free; they score a fixed set of brick poses against a
coarse IK reachability envelope and a table-collision check. Known-bad poses
are marked `xfail` and their failure reason is printed next to the result.

The ROS suite expects a sourced workspace; the core suite runs without
ROS as long as `requirements.txt` is installed.

## Extending

To add a new segmentation backend (e.g. SAM) alongside YOLO, follow
[`kinova_lego_assembly/core/segmentation/README.md`](kinova_lego_assembly/core/segmentation/README.md).
