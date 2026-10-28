#!/usr/bin/env bash
#
# Apply the vendored mujoco_ros2_control patch (see README.md in this directory).
#
# Usage:
#   patches/apply.sh [path-to-mujoco_ros2_control-repo]
#
# The target repo defaults to <workspace>/src/mujoco_ros2_control (resolved
# relative to this script) and can be overridden by the first argument or the
# MUJOCO_ROS2_CONTROL_DIR environment variable. The script is idempotent: if the
# patch is already applied it reports that and exits successfully.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/mujoco_ros2_control.patch"

# <ws>/src/bitbots_lego_assembly/src/kinova_mujoco_sim/patches -> <ws>
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
DEFAULT_TARGET="${WORKSPACE_ROOT}/src/mujoco_ros2_control"

TARGET="${1:-${MUJOCO_ROS2_CONTROL_DIR:-${DEFAULT_TARGET}}}"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "error: patch file not found: ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -d "${TARGET}/.git" ]]; then
  echo "error: '${TARGET}' is not a git repository." >&2
  echo "       Pass the mujoco_ros2_control checkout path as the first argument" >&2
  echo "       or set MUJOCO_ROS2_CONTROL_DIR." >&2
  exit 1
fi

echo "Patch : ${PATCH_FILE}"
echo "Target: ${TARGET}"

# Already applied? (the reverse patch would apply cleanly)
if git -C "${TARGET}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Patch already applied; nothing to do."
  exit 0
fi

# Verify it applies cleanly before touching the tree.
if ! git -C "${TARGET}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "error: patch does not apply cleanly to '${TARGET}'." >&2
  echo "       The vendored mujoco_ros2_control may be at a different revision" >&2
  echo "       than the patch base (see README.md)." >&2
  exit 1
fi

git -C "${TARGET}" apply "${PATCH_FILE}"
echo "Patch applied."
echo
echo "Rebuild (mujoco_ros2_control is pulled in as a dependency):"
echo "  colcon build --packages-up-to kinova_mujoco_sim --symlink-install"
echo "  (add --allow-overriding mujoco_ros2_control if the apt package is installed)"
