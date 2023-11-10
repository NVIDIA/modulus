
# import os
    
# run_dir = '/lustre/fsw/nvresearch/mmardani/output/logs/edm_cwb/private-frog_2022.12.19_15.31/code/training-runs/00002-cifar10-32x32-cond-ddpmpp-edm-gpus8-batch64-fp32'  #/kire-khar'

# files = os.listdir(run_dir)
# file_pkl = sorted([i for i in files if i.endswith('.pkl')])[-1:]
# file_pt = sorted([i for i in files if i.endswith('.pt')])[-1:]
# print(file_pkl)
# print(file_pt)

# if file_pkl and file_pt:
#     path_pkl = run_dir + file_pkl[0]
#     path_pt = run_dir + file_pt[0]
#     resume_kimg = int(file_pkl[0][17:-4])
#     print(path_pkl)
#     print(path_pt)
#     print(resume_kimg)
#     print(type(resume_kimg))
    
    
    
import os

path = "/lustre/fsw/nvresearch/mmardani/output/logs/edm_cwb/delicate-boa_2022.12.20_14.24/output/00000-cifar10-32x32-cond-ddpmpp-edm-gpus8-batch64-fp32"

tail = os.path.split(path)[0]

print(tail)