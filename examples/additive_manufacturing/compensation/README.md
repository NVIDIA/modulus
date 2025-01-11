

# PyTorch version of deformation predictor & compensation

## Key requirments

1. ``Torch_Geometric 2.5.1 or above``: PyTorch based geometric/graph neural network library
   
   - https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-via-anaconda
   
   - conda install pyg=*=*cu* -c pyg

2. ``pip install trimesh``

3. ``pip install matplotlib``

4. ``pip install pandas``

5. ``pip install hydra-core --upgrade --pre``

6. ``PyTorch3D``: PyTorch based 3D computer vision library 

   - Check requirements from official install page: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
   - when tested, Pytorch3D requires Python 3.8, 3.9 or 3.10
   - gcc --version
   
   - ``pip install -U iopath``
    
   - Install directly from the source ``pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" ``

7. ``pip install torch-cluster``


Other dependencies for development: 

- ``open3d``: pip install open3d, tested version 0.18.0
- ``torch-cluster``: conda install pytorch-cluster -c pyg



## Dataset
- Currently available: bar repository [link not working yet](https://duckduckgo.com)

- Sample input data folder format: 
  
   -  input_data.txt: logs for each row, the build geometry folder 
  
      - /part_folder_i:

         - cad_<part_id>.txt: contains 3 columns for point location 

         - scan_red<part_id>.csv: contains 3 columns for point location 

- Post-processing: 
  
    - https://github.azc.ext.hp.com/Shape-Compensation/Shape_compensator


## Training
- This is a two stage training analogous to GAN. DL deformation engine predicts part deformation. DL compensation engine propose a compensated shape, such that minimising the variation between the original part design and predicted printed part shape.   

- There are two training codes that need to run in sequential manner.
1. ``train_dis.py``: This code trains the discriminator (predict part deformations with its position and geometry) 
2. ``train_gen.py``: This code trains the generator (compensate part geometry)

## inference
- inference can be done with ``inference_engine.py`` 