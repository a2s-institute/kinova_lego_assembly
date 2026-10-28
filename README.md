# `kinova_lego_assembly`

A modular ROS 2 system that maintains a catalog of LEGO blueprints, receives a target model name at runtime, perceives the blocks in the workspace, and assembles the model on the LEGO holder with collision-aware motion. Validated in MuJoCo simulation, then deployed on the physical Kinova Gen3 arm.

<p align="center">
  <img src="docs/imgs/estop_sim.gif" alt="e_stop in MuJoCo simulation" height="260">
  &nbsp;
  <img src="docs/imgs/estop_real.gif" alt="e_stop on the real Kinova Gen3" height="260">
  &nbsp;
  <img src="docs/imgs/estop_rviz.gif" alt="e_stop in RViz with the camera stream" height="260">
</p>
<p align="center"><em><code>e_stop</code>: MuJoCo simulation (left), real robot (middle), and RViz with camera stream (right).</em></p>

---

## Repository layout

This repository is a colcon monorepo. The ROS 2 packages live under `src/`; shared tooling, assets, and docs live at the repository root.

### ROS 2 packages (`src/`)

| Package                                                                    | Role                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`kinova_lego_assembly`](src/kinova_lego_assembly)                       | The main ROS 2 application: LEGO perception, blueprint handling, and assembly orchestration. After workspace setup, follow its [README](src/kinova_lego_assembly/README.md) for package dependencies, build, and run instructions. |
| [`kinova_lego_assembly_interfaces`](src/kinova_lego_assembly_interfaces) | The ROS 2 messages, services, and actions used to communicate between application components.                                                                                                                                       |
| [`kinova_mujoco_sim`](src/kinova_mujoco_sim)                               | The MuJoCo simulation of the Kinova Gen3 arm, Robotiq gripper, table, holder, and LEGO bricks. Its [README](src/kinova_mujoco_sim/README.md) covers simulator-specific setup and usage.                                             |

The simulator also depends on the DFKI **`mujoco_ros2_control`** package, which is not released for our use case. It is vendored: cloned into the workspace at a pinned commit and patched. See [`src/kinova_mujoco_sim/patches/`](src/kinova_mujoco_sim/patches) for the patch, the apply script, and build instructions.

The brick / holder meshes, the MuJoCo `MjSpec` builders (`assets/mujoco/duplo_brick.py`, `assets/mujoco/lego_holder.py`), and the `lego_blueprints.yaml` catalog live in [`src/kinova_lego_assembly/assets/`](src/kinova_lego_assembly/assets). The simulator's scene generator imports the builders from there.

### Repo-root tooling (not owned by a ROS package)

- **`scripts/`**
  - `lint_blueprints.py`: schema validator for `src/kinova_lego_assembly/assets/lego_blueprints.yaml`.
  - `smoke_test_arm_controller.py`: commands the arm through a set of poses and logs the command / feedback exchange.
- **`ruff.toml`**: lint configuration.

---

## Documentation

| Topic                                   | Reference                                                        |
| --------------------------------------- | ---------------------------------------------------------------- |
| Simulator package (build, run, cameras) | [`kinova_mujoco_sim/README.md`](src/kinova_mujoco_sim/README.md) |
| Simulator scenes and layout knobs       | [`SCENES.md`](src/kinova_mujoco_sim/SCENES.md)                   |

---

## Getting started

### Prerequisites

| Requirement | Version          | Notes                |
| ----------- | ---------------- | -------------------- |
| Ubuntu      | 22.04 LTS        | Tested platform      |
| ROS 2       | Humble Hawksbill | `ros-humble-desktop` |
| Python      | 3.10             | Comes with Humble    |

### Create a workspace and clone the repository

```bash
mkdir -p ~/kinova_lego_assembly_ws/src
cd ~/kinova_lego_assembly_ws/src
git clone git@github.com:a2s-institute/kinova_lego_assembly.git
git clone git@github.com:Jay-droid/mas_industrial_robotics.git

# Return to the workspace root and set up a venv that can access ROS packages.
cd ~/kinova_lego_assembly_ws
sudo apt update
sudo apt install -y python3-venv python3-vcstool

source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages venv
touch venv/COLCON_IGNORE
source venv/bin/activate
```

The `mas_industrial_robotics` repository is a shared workspace dependency:
the application uses its `mir_interfaces` package, and the orchestrator uses
its MMOR object-recognition and pose-estimation packages.

### Next steps

- **Required: set up and build the main application package.** This package is
  required for running the system, including the simulator. Follow the
  [application package README](src/kinova_lego_assembly/README.md) for its
  dependencies, build, and run instructions.
- **For using the MuJoCo simulator:** follow the
  [simulation README](src/kinova_mujoco_sim/README.md) for its additional
  dependency and simulator-specific instructions.
