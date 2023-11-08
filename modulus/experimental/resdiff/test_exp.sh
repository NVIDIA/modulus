
# #prepare data (zip)
# python dataset_tool.py --source=/home/mmardani/research/datasets/cifar-10-python.tar.gz   --dest=/home/mmardani/research/datasets/cifar10-32x32.zip

# #test train run
# torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs   --data=/home/mmardani/research/datasets/cifar10-32x32.zip  --cond=1 --arch=ddpmpp   --batch 512


# #SELENE
# conda env create -f environment.yml -n edm
# conda activate edm

# pip install --ignore-installed Pillow==9.3.0
# /lustre/fsw/nvresearch/mmardani/source/weather-forecast-v2
# wget https://www.cs.toronto.edu/~kriz/cifar.html
# exec python dataset_tool.py --source=$SHARE_DATA/cifar10/cifar-10-python.tar.gz   --dest=$SHARE_DATA/cifar10/cifar10-32x32.zip


