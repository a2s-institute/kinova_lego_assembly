"""kinova_lego_assembly package init.

Sets process-wide environment defaults that must be in place before any
third-party dependency (notably Ultralytics) imports. ``os.environ.setdefault``
preserves explicit user overrides.
"""

import os

# Block Ultralytics' implicit ``pip install`` calls. Runtime deps are
# pinned via requirements.txt; surprise installs mask version drift and
# break offline / containerised deployments.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
