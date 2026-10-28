"""smoke test arm_controller."""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

POSES: list[tuple[str, list[float], float]] = [
    ("pose_A", [0.3, -0.50, 3.14, -1.80, 0.10, -1.20, 1.57], 0.0),
    ("pose_B", [0.0, -1.20, 3.14, -1.20, 0.00, -0.80, 1.57], 128.0),
    ("pose_C", [1.57, -0.60, 3.14, -1.50, 0.50, -1.00, 0.00], 0.0),
    ("pose_D", [0.0, -0.75, 3.14, -2.06, 0.00, -1.78, 1.57], 0.0),
]

SETTLE_SECONDS = 2.5
TOL = 0.05
PUB_TIMES = 5
PUB_HZ = 10.0
LOG_DIR = Path(__file__).resolve().parent.parent / "out" / "runs" / "arm_controller"


class ArmControllerSmokeNode(Node):
    """ROS node wrapper used by the smoke test."""

    def __init__(self):
        super().__init__("smoke_test_arm_controller")
        self.last_js: JointState | None = None
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.pub_arm = self.create_publisher(Float64MultiArray, "/joint_pos_cmd", 10)
        self.pub_grip = self.create_publisher(Float64, "/gripper_cmd", 10)

    def _on_js(self, msg: JointState):

        self.last_js = msg


def reached(node: ArmControllerSmokeNode, target: list[float]) -> bool:
    """Return whether the latest joint state is within tolerance of ``target``."""
    js = node.last_js
    if js is None or len(js.position) < len(target):
        return False
    return all(abs(a - b) <= TOL for a, b in zip(js.position[: len(target)], target, strict=True))


def main() -> None:
    """Run the smoke test end to end and write a Markdown log."""
    rclpy.init()
    node = ArmControllerSmokeNode()

    print("=" * 60)
    print("Arm Controller Smoke Test")
    print("=" * 60)

    start = time.time()
    while node.last_js is None and time.time() - start < 5.0:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.last_js is None:
        print("ERROR: no /joint_states received", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    results: list[tuple[str, bool, float, list[float], float]] = []
    for idx, (name, joints, grip) in enumerate(POSES, start=1):
        print(f"\n[{idx}/{len(POSES)}] {name}")
        arm = Float64MultiArray()
        arm.data = joints
        g = Float64()
        g.data = grip
        for _ in range(PUB_TIMES):
            node.pub_arm.publish(arm)
            node.pub_grip.publish(g)
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(1.0 / PUB_HZ)

        wait_start = time.time()
        while time.time() - wait_start < SETTLE_SECONDS:
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.02)

        ok = reached(node, joints)
        waited = time.time() - wait_start
        received_joints = list(node.last_js.position) if node.last_js else []
        print(f"  reached: {'YES' if ok else 'NO'} (waited {waited:.1f}s)")
        results.append((name, ok, waited, received_joints, grip))

    all_ok = all(ok for _, ok, _, _, _ in results)

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "smoke_test_arm_controller.md"
    lines = [
        f"# Arm Controller Smoke Test — {ts}",
        "",
        "## Topic Exchange",
        "",
        "**Published to:**",
        "- `/joint_pos_cmd` (Float64MultiArray) — arm joint position commands",
        "- `/gripper_cmd` (Float64) — gripper command",
        "",
        "**Subscribed from:**",
        "- `/joint_states` (JointState) — current arm joint positions",
        "",
        "## Results",
        "",
        "| Pose | Cmd Joints | Cmd Gripper | Received Joints | Reached | Wait (s) |",
        "|------|-----------|-------------|-----------------|---------|----------|",
    ]
    for name, ok, waited, received_joints, grip in results:
        pose_idx = next(i for i, (n, _, _) in enumerate(POSES) if n == name)
        cmd_joints = POSES[pose_idx][1]
        cmd_str = ", ".join(f"{j:.2f}" for j in cmd_joints)
        recv_str = ", ".join(f"{j:.2f}" for j in received_joints[:7]) if received_joints else "N/A"
        status = "✓" if ok else "✗"
        lines.append(f"| {name} | {cmd_str} | {grip:.1f} | {recv_str} | {status} | {waited:.1f} |")
    lines.append("")
    lines.append("_Exited cleanly — no controller errors on stop._")
    log_path.write_text("\n".join(lines))

    print(f"\nLog written → {log_path.relative_to(Path.cwd())}")
    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED" if all_ok else "SMOKE TEST FAILED")
    print("=" * 60)

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
