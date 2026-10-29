# Scenes and layout knobs

How to launch different tabletop layouts in `kinova_mujoco_sim` and which knobs
control the fixed work cell.

## Launching a scene

The `scene` launch argument selects which layout populates the table. It
defaults to `loose_pile`, and the chosen layout is deterministic: the same
bricks spawn in the same poses on every launch.

Source the workspace install space and launch as usual:

```bash
# default scene
ros2 launch kinova_mujoco_sim launch_mujoco_ros2_control.launch.py

# explicit scene
ros2 launch kinova_mujoco_sim launch_mujoco_ros2_control.launch.py scene:=loose_pile
```

An unknown name fails fast with `unknown scene '<name>'; choose from [...]`,
listing the registered scenes.

### Available scenes

| Scene                     | Content                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `loose_pile` (default)    | A loose assortment of 2x2 / 4x2 bricks laid flat in front-left of the holder. The holder itself stays empty.                          |
| `loose_pile_holder_stack` | The `loose_pile` plus a holder stack: a 4x2 then a 2x2 dropped onto the holder studs, with a flush neighbour 4x2/2x2 stack alongside. |

## How a scene fits together

Scene construction is two layers (see ADR-0013):

```text
build_scene(asset_root, scene_name)
  +-- build_stage(asset_root)            fixed furniture -> StageContext
  +-- scenes.get(scene_name).populate()  movable bricks, placed via ctx
  +-- spec.compile() -> (MjModel, MjData, StageContext)
```

- The **stage** (table, work arm, obstacle arm, welded holder, lights, camera)
  stays put across every scene. It is defined by `core/scene/layout.py` and built
  in `core/scene/stage.py`.
- A **scene** only adds movable brick bodies, placed relative to the stage
  anchors in `StageContext`, never hard-coded world coordinates.

`StageContext` gives a scene these anchors:

| Field                  | Meaning                                                                         |
| ---------------------- | ------------------------------------------------------------------------------- |
| `table_top_z`          | World height of the table top; loose bricks rest here.                          |
| `work_origin`          | World `(x, y, z)` the scene builds around (holder centre on the table top).     |
| `holder_pos`           | World position of the holder body origin.                                       |
| `holder_stud_top_z`    | World height of the holder stud tops; bricks dropped onto the holder land here. |
| `obstacle_joint_names` | Prefixed names of the obstacle arm's seven joints.                              |

## Adding a new scene

1. Create `kinova_mujoco_sim/core/scene/scenes/<your_scene>.py`.
2. Implement the `Scene` protocol (a `name` plus a `populate` method) and
   register an instance:

```python
from kinova_mujoco_sim.core.scene.scenes.base import register
from kinova_mujoco_sim.core.scene.stage import StageContext
import mujoco


class SingleBrickScene:
    name = "single_brick"

    def populate(self, spec: mujoco.MjSpec, ctx: StageContext) -> None:
        from assets.mujoco.duplo_brick import add_duplo_brick, _BRICK_HALF_HEIGHT_M

        wx, wy, _ = ctx.work_origin
        rest_z = ctx.table_top_z + _BRICK_HALF_HEIGHT_M
        add_duplo_brick(
            spec,
            brick_type="2x2",
            colour="red",
            name="solo",
            pos=(wx, wy, rest_z),
        )


register(SingleBrickScene())
```

3. Launch it with `scene:=single_brick`. The scenes package auto-imports every
   module in `core/scene/scenes/` (except `base`) on import, so dropping the file
   in is enough to register it; no `__init__.py` edit is needed.

Keep layouts deterministic: fixed poses, no random sampling. Use
`kinova_mujoco_sim.core.utils.transforms.yaw_quat(deg)` for flat rotations rather
than hand-rolled quaternions.

## Important knobs

### Fixed layout (`core/scene/layout.py`)

| Constant                      | Controls                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `TABLE_DIMENSIONS`            | `(length_x, width_y, thickness_z)` of the legless table box.                                           |
| `TABLE_TOP_Z`                 | Height of the table top plane.                                                                         |
| `TABLE_RGBA`                  | Table colour (currently `#b1b7df`).                                                                    |
| `ARM_INSET_FROM_BACK_EDGE_X`  | How far the arm bases sit in front of the back edge.                                                   |
| `ARM_Y_SEPARATION`            | Centre-to-centre spacing of the two arms along y.                                                      |
| `ARM_MOUNT_PLATE_THICKNESS_Z` | Riser height under each arm base, so the table plane sits 25 mm below `base_link` as on the real cell. |
| `ARM2_QPOS`                   | Default static pose of the obstacle arm.                                                               |
| `HOLDER_POS_XY`               | Holder centre on the table plane.                                                                      |
| `HOLDER_YAW_DEG`              | Holder rotation about the world z-axis.                                                                |
| `CONTACT_SOLREF`              | Contact stiffness so dropped bricks do not sink.                                                       |

Arm anchors are derived from the table edge, so the arms stay on the table for
any `TABLE_DIMENSIONS`.

### Obstacle arm

The second arm is the same `gen3.xml` attached with an `obs_` prefix; nothing
wires it to ros2_control. The MJCF generator writes an initial keyframe that
sets the obstacle joints and their stiff position actuators' `ctrl` to
`layout.ARM2_QPOS`, so the arm holds that static configuration as a passive
collision obstacle. Change the pose by editing `layout.ARM2_QPOS`.

## Known limitations

- The brick / holder builders are imported through a `sys.path` shim to
  `src/kinova_lego_assembly/assets/mujoco`. Proper packaging is deferred work.
- Physics runs inside the `mujoco_ros2_control` plugin (the MuJoCo step loop is
  driven by the controller manager), not the passive viewer. Headless runs are
  supported (`show_gui:=false`); shut the simulation down with `Ctrl+C` in the
  launch terminal. Closing the MuJoCo GUI window does not stop the nodes.
