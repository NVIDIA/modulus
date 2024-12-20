# Training recipes for External Aerodynamics

![Results from DoMINO for RTWT SC demo](../../../../docs/img/domino_result_rtwt.jpg)

We have a set of model training recipes that highlight key architectures for external automotive aerodynamics use case. 
The architectures are:
- DoMINO is a local, multi-scale, point-cloud based model architecture and the recipe trains a model to take STL
geometries as input and evaluates flow quantities such as pressure and wall shear stress on the surface of the car as well as velocity fields and pressure in the volume around it. Learn more about training recipe with DoMINO [here](https://github.com/NVIDIA/modulus/blob/ram-cherukuri-patch-1/examples/cfd/external_aerodynamics/domino/README.md)).
- XAeroNet is a collection of scalable models for large-scale external aerodynamic evaluations. It consists of two models, XAeroNet-S and XAeroNet-V for surface and volume predictions, respectively. Learn more about training recipe with XAeroNet [here](https://github.com/NVIDIA/modulus/blob/ram-cherukuri-patch-1/examples/cfd/external_aerodynamics/xaeronet/README.md)
- FIGConvUNet, a novel architecture that can efficiently scale for large 3D meshes and arbitrary input and output geometries. FIGConvUNet efficiently combines U-shaped architecture, graph information gathering, and integration, learning efficient latent representation through the representation graph voxel layer. Learn more about training recipe with FIGConvUNet [here](https://github.com/NVIDIA/modulus/blob/ram-cherukuri-patch-1/examples/cfd/external_aerodynamics/figconvnet/README.md)
