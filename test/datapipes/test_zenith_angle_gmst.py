# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression test for the Greenwich mean sidereal time polynomial used by the
DALI ``cos_zenith_angle`` datapipe (``physicsnemo/datapipes/climate/utils/
zenith_angle.py``).

The module is loaded directly from source with a numpy-backed stub for its
optional ``nvidia.dali`` dependency, so the test exercises the real shipped
code without requiring DALI (or the rest of the physicsnemo package, which
pulls in heavy GPU dependencies at import time).
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# physicsnemo/datapipes/climate/utils/zenith_angle.py, relative to test/.
_SRC = (
    Path(__file__).resolve().parents[2]
    / "physicsnemo"
    / "datapipes"
    / "climate"
    / "utils"
    / "zenith_angle.py"
)


def _load_dali_zenith_module():
    """Load the real DALI zenith-angle source with a numpy-backed dali stub."""
    # Stub ``nvidia.dali``: the module only needs ``dali.math.floor`` for the
    # modulo used by the sidereal-time computation.
    dali_stub = types.SimpleNamespace(math=types.SimpleNamespace(floor=np.floor))

    class _OptionalImport:
        def __new__(cls, module_name, *args, **kwargs):
            return dali_stub

    # Provide a minimal physicsnemo.core.version_check so the module's
    # ``from physicsnemo.core.version_check import OptionalImport`` resolves
    # without importing the full package.
    pkg = types.ModuleType("physicsnemo")
    pkg.__path__ = []  # mark as package
    core = types.ModuleType("physicsnemo.core")
    core.__path__ = []
    vc = types.ModuleType("physicsnemo.core.version_check")
    vc.OptionalImport = _OptionalImport

    saved = {
        name: sys.modules.get(name)
        for name in (
            "physicsnemo",
            "physicsnemo.core",
            "physicsnemo.core.version_check",
        )
    }
    sys.modules["physicsnemo"] = pkg
    sys.modules["physicsnemo.core"] = core
    sys.modules["physicsnemo.core.version_check"] = vc
    try:
        spec = importlib.util.spec_from_file_location("_dali_zenith_under_test", _SRC)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return module


def _reference_gmst(module, timestamp):
    """Greenwich mean sidereal time (radians) from the AIAA 2006 polynomial.

    This mirrors the repository's torch implementation in
    ``physicsnemo/utils/zenith_angle.py`` (cubic coefficient ``6.2e-6``).
    """
    days = (timestamp - module.DATETIME_2000) / (24.0 * 3600.0)
    jul_centuries = days / 36525.0
    theta = 67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2e-6)
    )
    return (theta / 240.0 * module.RAD_PER_DEG) % (2 * np.pi)


# Seconds since 2000-01-01 12:00:00 UTC for a few epochs. The sidereal cubic
# term scales with (Julian centuries)**3, so a date well away from the 2000
# epoch is needed for that term to rise above float64 noise.
_TIMESTAMPS = [
    0.0,  # 2000-01-01 12:00 UTC
    7.9e8,  # ~2025
    6.3e9,  # ~2200
]


@pytest.mark.parametrize("timestamp", _TIMESTAMPS)
def test_dali_gmst_matches_aiaa_reference(timestamp):
    module = _load_dali_zenith_module()
    actual = module._greenwich_mean_sidereal_time(timestamp)
    expected = _reference_gmst(module, timestamp)
    # After the fix the two polynomials are identical, so they agree exactly;
    # the buggy 10x cubic coefficient drives them apart by far more than atol.
    assert np.isclose(actual, expected, rtol=0.0, atol=1e-9), (
        f"GMST mismatch at t={timestamp}: {actual!r} vs reference {expected!r}"
    )
