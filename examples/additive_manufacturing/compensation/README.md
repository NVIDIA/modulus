

# PyTorch version of deformation predictor & compensation

## Key requirments

1. Install Python version compatible

   - when tested, Pytorch3D requires Python 3.8, 3.9 or 3.10

4. ``PyTorch3D``: PyTorch based 3D computer vision library 

   - Check requiremenets from official install page: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
   
   - gcc --version
   
   - pip3 install -U fvcore iopath
   
   - conda install -c bottler nvidiacub
   
   - Install directly from the source ``pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" ``
   
1. Check Cuda version 
   
   - nvcc -version 
   
3. Install matching PyTorch 

   - when tested, Pytorch3D requires Pytorch 1.12.0, 1.12.1, 1.13.0, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2 or 2.2.0.
   
   - Official installation or earlier cuda versions: https://pytorch.org/get-started/previous-versions/
   
   - i.e. conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
   
6. ``Torch_Geometric 2.5.1 or above``: PyTorch based geometric/graph neural network library
   
   - https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-via-anaconda
   
   - conda install pyg=*=*cu* -c pyg
   
4. ``trimesh``: pip install trimesh, tested version ....

5. ``open3d``: pip install open3d, tested version 0.18.0

7. ``matplotlib``: pip install matplotlib

8. ``pandas``: pip install pandas

9. ``torch-cluster``: conda install pytorch-cluster -c pyg





## Dataset
- currently only support Jordi's bar repository [link not working yet](https://duckduckgo.com)

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