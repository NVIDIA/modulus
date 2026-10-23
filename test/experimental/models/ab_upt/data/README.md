# AB-UPT regression fixtures

These fixtures were generated from this independent PhysicsNeMo implementation,
not the research repository. They check serialization and numerical regressions;
they do not establish agreement with research weights or benchmark accuracy.

The two configurations in `test_ab_upt.py` use seeds 1729 and 1730, respectively.
Each `.mdlus` contains the untrained model; `_output.pth` contains its forward
outputs for the standard model-test helper; `_methods.pth` contains inputs and
reference geometry encodings, anchor contexts, query outputs, and vorticity.
The fixtures were produced on CPU with PyTorch 2.14.0 in float32.
