# Lid Driven Cavity Flow using Purely Physics Driven Neural Networks (PINNs)

This example demonstrates how to set up a purely physics-driven model for solving a Lid
Driven Cavity (LDC) flow using PINNs. The goal of this example is to demonstrate the
interoperability of Modulus, Modulus-Sym and PyTorch. This example adopts a workflow
where appropriate utilities are imported from `modulus`, `modulus.sym` and `torch` to
define the training pipeline.

Specifically, this example demonstrates how the geometry and physics utilites from
Modulus-Sym can be used in custom training pipelines to handle geometry objects
(typically found in Computer Aided Engineering (CAE)) workflows and introduce physics
residual and boundary condition losses.

This example takes a non-abstracted way to define the problem. The
boundary condition constraints, residual constraints, and the subsequent physics loss
computation are defined explicitly. For a more abstracted version of this workflow,
where some of these steps are automated and abstracted, we recommend the
[Introductory example tutorial from Modulus-Sym](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/basics/lid_driven_cavity_flow.html).

## Getting Started

### Prerequisites

If you are running this example outside of the Modulus container, install Modulus Sym using
the instructions from [here](https://github.com/NVIDIA/modulus-sym?tab=readme-ov-file#pypi)

### Training

To train the model, run

```bash
python train.py
```

This should start training the model. Since this is training in a purely Physics based
fashion, there is no dataset required.

Instead, we generate the geometry using the Modulus Sym's geometry module and sample
point cloud using `GeometryDatapipe` utility. For more details refer documentation
[here](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/api/modulus.sym.geometry.html#modulus.sym.geometry.geometry_dataloader.GeometryDatapipe)

For computing the physics losses, we will use the `PhysicsInformer` utility from
Modulus-Sym. For more details, refer documentation
[here](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/api/modulus.sym.eq.html#modulus.sym.eq.phy_informer.PhysicsInformer)

The results would get saved in the `./outputs/` directory.

## Additional Reading

This example demonstrates computing physics losses on point clouds. For more examples
on physics informing different type of models and model outputs, refer:

* Point clouds: [Darcy Flow (DeepONet)](../darcy_physics_informed/darcy_physics_informed_deeponet.py),
[Stokes Flow (MLP)](../stokes_mgn/pi_fine_tuning.py)
* Regular grid: [Darcy Flow (FNO)](../darcy_physics_informed/darcy_physics_informed_fno.py)
* Unstructured meshes: [Stokes Flow (MeshGraphNet)](../stokes_mgn/pi_fine_tuning_gnn.py)
