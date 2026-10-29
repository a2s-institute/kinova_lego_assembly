# kinova_lego_assembly

Package providing LEGO segmentation and assembly nodes.

Setup, build, and launch instructions are in the
[root README](../../README.md#getting-started).

## Tests

Tests are split between pure-Python (`test/core/`) and ROS-dependent
(`test/ros/`) suites, with shared assets under `test/assets/`.

List everything [pytest](https://docs.pytest.org/) would collect (no execution):

```bash
cd ~/kinova_lego_assembly_ws
cd src/kinova_lego_assembly
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
