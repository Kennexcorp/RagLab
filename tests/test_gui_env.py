"""
Tests for process-level environment invariants the GUI depends on.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_gui_app_selects_numba_workqueue_before_numba_is_imported():
    """gui_app must pin numba's threading layer before anything imports numba.

    numba's default layer selection picks OpenMP, and torch ships its own OpenMP
    runtime. With both loaded, numba's parallel regions segfault (SIGSEGV) inside
    UMAP.transform() — which is exactly what the Explore tab does after embedding a
    query with torch. Setting NUMBA_THREADING_LAYER=workqueue avoids the second
    runtime, but only takes effect if it is set before numba is first imported.
    """
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_THREADING_LAYER"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import gui_app, os, sys; "
            "print(os.environ.get('NUMBA_THREADING_LAYER'), 'numba' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, f"importing gui_app failed:\n{result.stderr}"
    layer, numba_already_imported = result.stdout.split()[-2:]
    assert layer == "workqueue"
    assert numba_already_imported == "False", (
        "numba was imported during gui_app import, so NUMBA_THREADING_LAYER was set too late"
    )
