# Segmentation backends

The segmentation core is intentionally backend-agnostic. The
[`LegoSegmentation` ROS node](../../ros/nodes/lego_segmentation.py)
only talks to [`build_segmentor()`](factory.py) and re-declares ROS
parameters from whatever knobs every registered backend exposes, so
adding a new backend (e.g. SAM, Mask R-CNN, a classical CV pipeline)
does not touch the node at all.

## Adding a backend

Four small changes, all isolated to this subtree plus one config file.

1. **Implement [`SegmentationModel`](base.py).** Create
   `<method>_segmentor.py` next to
   [`yolo_segmentor.py`](yolo_segmentor.py) and implement
   `class_names`, `device`, `infer(image) -> SegmentationResult`, and
   (only if you have tunable knobs) `update_cfg(**overrides)`. Decode
   each detection into a [`LegoObject`](../types/vision.py); fields
   you cannot fill (e.g. a backend without polygons) stay `None`.
   Expose the tunable knob defaults as a `ClassVar[dict[str, Any]]`
   named `DEFAULTS` so the factory can advertise them.

2. **Register the backend in [`factory.py`](factory.py).** Add a
   `_<method>_spec()` builder and an entry under `_REGISTRY`:

   ```python
   def _sam_spec() -> BackendSpec:
       return BackendSpec(
           knobs=dict(SAMSegmentor.DEFAULTS),
           build=lambda version=None, **overrides: SAMSegmentor(version=version, **overrides),
       )


   _REGISTRY: dict[str, Callable[[], BackendSpec]] = {
       "yolo": _yolo_spec,
       "sam": _sam_spec,
   }
   ```

3. **Declare the knob group in
   [`config/params.yml`](../../../config/params.yml).** Add a
   `segmentation.<method>:` block whose keys match
   `<method>Segmentor.DEFAULTS`. The node iterates every registered
   backend's knobs at startup, so the `segmentation.<method>.<knob>`
   parameters are then live-tunable through `ros2 param set` without
   any node changes.

4. **Drop weights under
   [`trained_models/`](../../../trained_models/README.md).** Use the
   layout `trained_models/<task>/<method>/v<N>/`, with a `model.yaml`
   that maps `class_id -> class_name`. The
   [registry](registry.py) auto-picks the highest `v<N>`; override
   with the `segmentation.model_version` ROS parameter.

Switching the active backend at deploy time is then just
`segmentation.model_method: sam` in `params.yml`, no code or rebuild
required.

## Configuration sources

The core and the ROS layer pull defaults from different places, by
design:

| Caller | Defaults | Overrides |
|---|---|---|
| Pure Python (notebooks, `pytest`, scripts) | `<method>Segmentor.DEFAULTS` | kwargs to `build_segmentor()` / `update_cfg()` |
| ROS service node | [`config/params.yml`](../../../config/params.yml) overlaid onto `DEFAULTS` | `ros2 param set segmentation.<method>.<knob>` |

[`config/params.yml`](../../../config/params.yml) is a ROS-only
artefact (its `ros__parameters` schema is parsed by `rclpy`, not by
the core) and is **not** read by pure-Python callers. Keep its
`<method>:` block numerically in sync with `<method>Segmentor.DEFAULTS`
so the two paths agree.

## Test layout

Mirror the existing split: a pure-Python integration suite under
[`test/core/`](../../../test/core) (loads weights, runs `infer()`,
asserts the decoded `SegmentationResult`) and, if the backend has
ROS-specific surface, a thin smoke test under
[`test/ros/`](../../../test/ros).
