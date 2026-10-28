# `kinova_lego_assembly`

**ASD Lab Summer Semester 2026 · Team 2 · CW20–CW28**

A modular ROS 2 system that maintains a catalog of LEGO blueprints, receives a target model name at runtime, perceives the blocks in the workspace, and assembles the model on the LEGO holder with collision-aware motion. Validated in MuJoCo simulation, then deployed on the physical Kinova Gen3 arm.

> Team: Scotrraaj · Jay · Suhail
> Supervised by: Mohammad Wasil · Gokul

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
| [`bitbots_lego_assembly`](src/bitbots_lego_assembly)                       | The main ROS 2 application: LEGO perception, blueprint handling, and assembly orchestration. After workspace setup, follow its [README](src/bitbots_lego_assembly/README.md) for package dependencies, build, and run instructions. |
| [`bitbots_lego_assembly_interfaces`](src/bitbots_lego_assembly_interfaces) | The ROS 2 messages, services, and actions used to communicate between application components.                                                                                                                                       |
| [`kinova_mujoco_sim`](src/kinova_mujoco_sim)                               | The MuJoCo simulation of the Kinova Gen3 arm, Robotiq gripper, table, holder, and LEGO bricks. Its [README](src/kinova_mujoco_sim/README.md) covers simulator-specific setup and usage.                                             |

The simulator also depends on the DFKI **`mujoco_ros2_control`** package, which is not released for our use case. It is vendored: cloned into the workspace at a pinned commit and patched. See [`src/kinova_mujoco_sim/patches/`](src/kinova_mujoco_sim/patches) for the patch, the apply script, and build instructions.

### Repo-root tooling (not owned by a ROS package)

These live at the repository root and are used across packages or standalone; they are documented here because they are not part of any package's README.

- **`assets/`** — MuJoCo `MjSpec` builders (`mujoco/duplo_brick.py`, `mujoco/lego_holder.py`) shared by the simulator's scene generator, plus brick / holder meshes and the `lego_blueprints.yaml` catalog.
- **`scripts/`**
  - `align_stl_axes.py` — normalize STL meshes (centre the bounding box, put the long edge on +X).
  - `blenderproc_lego_coco.py` — render a synthetic LEGO-Duplo COCO dataset via BlenderProc (one scene per invocation; driven by a notebook).
  - `launch_duplo_viewer.py` — standalone MuJoCo viewer for a small Duplo scene (no ROS), using the `assets/mujoco` builders.
  - `lint_blueprints.py` — schema validator for `assets/lego_blueprints.yaml` (run in CI; the spec is [ADR-0009](docs/adr/0009-blueprint-schema.md)).
  - `smoke_test_arm_controller.py` — commands the arm through a set of poses and logs the command / feedback exchange.
- **`notebooks/`** — exploratory and dataset-generation notebooks (e.g. the BlenderProc dataset driver).
- **`docs/`** — architecture decision records (`docs/adr/`) and the [release process](docs/release-process.md).
- **`ruff.toml`**, **`CONTRIBUTING.md`** — lint configuration and contributor workflow.

---

## Documentation

| Topic                                       | Reference                                                        |
| ------------------------------------------- | ---------------------------------------------------------------- |
| End-to-end LEGO assembly pipeline           | [`docs/assembly-pipeline.md`](docs/assembly-pipeline.md)         |
| Daily workflow, labels, branches, MRs       | [`CONTRIBUTING.md`](CONTRIBUTING.md)                             |
| Architecture decisions                      | [`docs/adr/`](docs/adr/)                                         |
| Simulator package (build, run, cameras)     | [`kinova_mujoco_sim/README.md`](src/kinova_mujoco_sim/README.md) |
| Simulator scenes and layout knobs           | [`SCENES.md`](src/kinova_mujoco_sim/SCENES.md)                   |
| Semester plan                               | [Milestone-planning][wiki-plan] (wiki)                           |
| Team ownership and per-milestone assignment | [team-strategy][wiki-team] (wiki)                                |
| Release process `devel` to `main`           | [`release-process.md`](docs/release-process.md)                  |

### Build the Sphinx documentation

From this repository's root, install the documentation dependencies and build the HTML pages:

```bash
python3 -m pip install -r docs/requirements.txt
python3 -m sphinx -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser to view the rendered documentation.

## Live status

- [Workflow board][board] — daily-standup view; per-milestone and per-assignee URL recipes are in [`CONTRIBUTING.md` §4](CONTRIBUTING.md#4-the-workflow-board).
- [Milestones][milestones] — native GitLab progress and burndown.
- [Issues][issues] · [Merge requests][mrs] — what's open and what just landed.

[board]: https://git.inf.h-brs.de/sdp-ss-2026/b-it-bots-at-work/bitbots_lego_assembly/-/boards/269
[milestones]: https://git.inf.h-brs.de/sdp-ss-2026/b-it-bots-at-work/bitbots_lego_assembly/-/milestones
[issues]: https://git.inf.h-brs.de/sdp-ss-2026/b-it-bots-at-work/bitbots_lego_assembly/-/issues
[mrs]: https://git.inf.h-brs.de/sdp-ss-2026/b-it-bots-at-work/bitbots_lego_assembly/-/merge_requests
[wiki-plan]: https://git.inf.h-brs.de/sdp-ss-2026/b-it-bots-at-work/bitbots_lego_assembly/-/wikis/Milestone-planning
[wiki-team]: https://git.inf.h-brs.de/sdp-ss-2026/b-it-bots-at-work/bitbots_lego_assembly/-/wikis/team-strategy

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
mkdir -p ~/bitbots_lego_assembly_ws/src
cd ~/bitbots_lego_assembly_ws/src
git clone https://git.inf.h-brs.de/sdp-ss-2026/b-it-bots-at-work/bitbots_lego_assembly.git
git clone git@github.com:Jay-droid/mas_industrial_robotics.git

# Return to the workspace root and set up a venv that can access ROS packages.
cd ~/bitbots_lego_assembly_ws
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
  [application package README](src/bitbots_lego_assembly/README.md) for its
  dependencies, build, and run instructions.
- **For using the MuJoCo simulator:** follow the
  [simulation README](src/kinova_mujoco_sim/README.md) for its additional
  dependency and simulator-specific instructions.

### Run lint checks locally (mirrors CI)

```bash
# Blueprint schema lint
python3 scripts/lint_blueprints.py

# Python lint (report only; matches CI)
ruff check

# Python lint with autofix applied
ruff check --fix

# Python formatting check (report only; matches CI)
ruff format --check

# Python formatting applied in place
ruff format
```
