
submit_job --image gitlab-master.nvidia.com/tkurth/era5_wind:latest --duration 4 --autoresume_timer 235 --nodes 2 --gpu 8 --mounts $SHARE_SOURCE,$SHARE_OUTPUT,$SHARE_DATA,/lustre/fsw --setenv SUBMIT_ACCOUNT=devtech --partition luna  --name exp-test --command ' cd $SHARE_SOURCE/weather-forecast-v2; PYTHONPATH=$SHARE_SOURCE/weather-forecast-v2 exec torchrun  --nnodes=2  --nproc_per_node=8  --max_restarts=3  --rdzv_endpoint=????   --rdzv_id=????  --rdzv_backend=c10d   train.py --batch 128 --arch ddpmpp --precond edm --data /lustre/fsw/sw_climate_fno/nbrenowitz/2022-01-19-cwb.zarr --outdir $SHARE_OUTPUT/logs/exp-test/output --lr 2e-4 --duration 200 --snap 2 --dump 2 --workers 4 --data_config full_field_train_crop112_grid_20inchans_4x --task sr --data_type era5-cwb-v2  ' --logdir $SHARE_OUTPUT/logs/exp-test/gcf_log


# WARNING:torch.distributed.run:
# *****************************************
# Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
# *****************************************
# INFO:torch.distributed.launcher.api:Starting elastic_operator with launch configs:
#   entrypoint       : train.py
#   min_nodes        : 2
#   max_nodes        : 2
#   nproc_per_node   : 8
#   run_id           : none
#   rdzv_backend     : c10d
#   rdzv_endpoint    : localhost:29500
#   rdzv_configs     : {'timeout': 900}
#   max_restarts     : 3
#   monitor_interval : 5
#   log_dir          : None
#   metrics_cfg      : {}

# INFO:torch.distributed.elastic.agent.server.local_elastic_agent:log directory set to: /tmp/torchelastic_967sd_f4/none_iqf20aq2
# INFO:torch.distributed.elastic.agent.server.api:[default] starting workers for entrypoint: python
# INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous'ing worker group
# Traceback (most recent call last):
#   File "/opt/conda/bin/torchrun", line 33, in <module>
#     sys.exit(load_entry_point('torch==1.13.0a0+d0d6b1f', 'console_scripts', 'torchrun')())
# INFO:torch.distributed.elastic.agent.server.local_elastic_agent:log directory set to: /tmp/torchelastic_967sd_f4/none_iqf20aq2
# INFO:torch.distributed.elastic.agent.server.api:[default] starting workers for entrypoint: python
# INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous'ing worker group
# Traceback (most recent call last):

#   File "/opt/conda/bin/torchrun", line 33, in <module>
#     sys.exit(load_entry_point('torch==1.13.0a0+d0d6b1f', 'console_scripts', 'torchrun')())
#   File "/opt/conda/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 345, in wrapper
#  File "/opt/conda/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 538, in _rendezvous
#     store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
#   File "/opt/conda/lib/python3.8/site-packages/torch/distributed/elastic/rendezvous/dynamic_rendezvous.py", line 1025, in next_rendezvous
#     self._op_executor.run(join_op, deadline)
#   File "/opt/conda/lib/python3.8/site-packages/torch/distributed/elastic/rendezvous/dynamic_rendezvous.py", line 638, in run
#     raise RendezvousTimeoutError()
# torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError


#error for nnodes=2
#vim job-3355891_1_luna-0467_stderr.log


#--rdzv_id

# INFO:torch.distributed.launcher.api:Starting elastic_operator with launch configs:
#   entrypoint       : train.py
#   min_nodes        : 1
#   max_nodes        : 1
#   nproc_per_node   : 8
#   run_id           : none
#   rdzv_backend     : c10d
#   rdzv_endpoint    :
#   rdzv_configs     : {'timeout': 900}
#   max_restarts     : 3
#   monitor_interval : 5
#   log_dir          : None
#   metrics_cfg      : {}

# INFO:torch.distributed.elastic.agent.server.local_elastic_agent:log directory set to: /tmp/torchelastic_b2ohx378/none_prli743_
# INFO:torch.distributed.elastic.agent.server.api:[default] starting workers for entrypoint: python
# INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous'ing worker group
# INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous complete for workers. Result:
#   restart_count=0
#   master_addr=luna-0467.selene.nvidia.com
#   master_port=55341
#   group_rank=0
#   group_world_size=1
#   local_ranks=[0, 1, 2, 3, 4, 5, 6, 7]
#   role_ranks=[0, 1, 2, 3, 4, 5, 6, 7]
#   global_ranks=[0, 1, 2, 3, 4, 5, 6, 7]
#   role_world_sizes=[8, 8, 8, 8, 8, 8, 8, 8]
#   global_world_sizes=[8, 8, 8, 8, 8, 8, 8, 8]

# INFO:torch.distributed.elastic.agent.server.api:[default] Starting worker group



# os.environ environ({'DALI_BUILD': '5838886', 'SLURM_STEP_NODELIST': 'luna-0197', 'LESSOPEN': '| /usr/bin/lesspipe %s', 'LIBRARY_PATH': '/usr/local/cuda/lib64/stubs:', 'PYTORCH_BUILD_NUMBER': '0', 'PYTHONIOENCODING': 'utf-8', 'SLURM_JOB_USER': 'mmardani', 'SLURM_CPU_BIND_LIST': '', 'SLURM_JOBID': '3355847', 'USER': 'root', 'PYTORCH_HOME': '/opt/pytorch/pytorch', 'SSH_CLIENT': '10.110.38.243 53432 22', 'SLURM_PTY_PORT': '33595', 'MASTER_ADDR': 'luna-0197.selene.nvidia.com', 'CUSOLVER_VERSION': '11.4.1.48', 'SRUN_DEBUG': '3', 'SLURM_JOB_QOS': 'normal', 'COCOAPI_VERSION': '2.0+nv0.6.1', 'CUTENSOR_VERSION': '1.6.1.5', 'PMIX_RANK': '0', 'SLURM_JOB_NUM_NODES': '1', 'SLURM_SRUN_COMM_PORT': '40571', 'SHLVL': '1', 'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib.real:/opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64', 'SLURM_TASKS_PER_NODE': '1', 'MASTER_PORT': '40101', 'HOME': '/root', 'PMIX_MCA_psec': 'none', 'PMIX_NAMESPACE': 'slurm.pmix.3355847.0', 'OLDPWD': '/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr_fulldata/holistic-falcon_2023.01.29_13.49', 'NVIDIA_PIPELINE_ID': '5985389', 'PMIX_GDS_MODULE': 'hash', 'PMIX_SYSTEM_TMPDIR': '/var/empty', 'SLURM_TOPOLOGY_ADDR_PATTERN': 'switch.switch.switch.node', 'NVM_BIN': '/usr/local/nvm/versions/node/v16.15.1/bin', 'SSH_TTY': '/dev/pts/48', 'NVM_INC': '/usr/local/nvm/versions/node/v16.15.1/include/node', 'NCCL_P2P_READ_ENABLE': '1', 'CUDA_CACHE_DISABLE': '1', 'SLURM_CPU_BIND_TYPE': 'none', 'SHARE_DATA': '/lustre/fsw/nvresearch/mmardani/data', 'SLURM_WORKING_CLUSTER': 'selene:10.248.1.196:6817:9216:199', 'SLURM_PRIO_PROCESS': '0', 'WORLD_SIZE': '8', 'SLURM_JOB_CPUS_PER_NODE': '256', 'ENV': '/etc/shinit_v2', 'RDMACORE_VERSION': '36.0', 'NVJPEG_VERSION': '11.9.0.86', 'NVIDIA_BUILD_ID': '44877844', 'OMPI_MCA_btl_openib_warn_default_gid_prefix': '0', 'SLURM_JOB_NAME': 'devtech-e2prep:interactive', 'SLURM_JOB_GID': '30', 'SLURM_CPUS_ON_NODE': '256', 'SLURM_PROCID': '0', 'CUDA_VERSION': '11.8.0.062', 'NVM_DIR': '/usr/local/nvm', 'SLURM_JOB_ACCOUNT': 'devtech', 'RANK': '0', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1', 'CUBLAS_VERSION': '11.11.3.6', 'TORCH_CUDA_ARCH_LIST': '5.2 6.0 6.1 7.0 7.5 8.0 8.6 9.0+PTX', 'CUDA_VIRTUAL': '', 'NSIGHT_SYSTEMS_VERSION': '2022.3.1.43', 'TMPDIR': '/tmp', 'PMIX_SERVER_URI2': 'pmix-server.971726;tcp4://127.0.0.1:36747', 'PMIX_MCA_gds': 'hash', 'SLURM_CONF': '/etc/slurm/slurm.conf', 'SLURM_STEP_LAUNCHER_PORT': '40571', 'LOGNAME': 'mmardani', 'CUBLAS_VIRTUAL': '', 'PMIX_SERVER_URI3': 'pmix-server.971726;tcp4://127.0.0.1:36747', 'OPAL_PREFIX': '/opt/hpcx/ompi', 'CUDA_MODULE_LOADING': 'LAZY', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'TRT_VERSION': '8.5.0.12', 'GDRCOPY_VERSION': '2.3', 'SUBMIT_LOGS': '/lustre/fsw/adlr/adlr-nlp/mmardani//logs', 'SLURM_SUBMIT_HOST': 'selene-login-01', 'PYTORCH_BUILD_VERSION': '1.13.0a0+d0d6b1f', 'SLURM_MPI_TYPE': 'pmix', '_': '/opt/conda/bin/python3', 'RUNTESTS_LOGS_SELENE': '/lustre/fsw/adlr/adlr-nlp/mmardani//runtests_logs', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility,video', 'POLYGRAPHY_VERSION': '0.38.0', 'CURAND_VERSION': '10.3.0.86', 'TRT_VIRTUAL': '', 'MOFED_VERSION': '5.4-rdmacore36.0', 'TERM': 'xterm-256color', 'SLURM_NODELIST': 'luna-0197', 'SLURM_PMIX_MAPPING_SERV': '(vector,(0,1,1))', 'SLURM_NNODES': '1', 'SLURM_JOB_ID': '3355847', 'CLUSTER': 'SELENE', 'SLURM_CPU_BIND': 'quiet,none', 'SLURMD_NODENAME': 'luna-0197', 'PMIX_VERSION': '3.1.5', 'TORCH_CUDNN_V8_API_ENABLED': '1', 'NVIDIA_PYTORCH_VERSION': '22.09', 'SLURM_JOB_NODELIST': 'luna-0197', 'PATH': '/lustre/fsw/adlr/adlr-others/gpeled/adlr-utils/release/cluster-interface/latest:/usr/local/nvm/versions/node/v16.15.1/bin:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/bin:/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin', 'SLURM_GTIDS': '0', 'SLURM_STEPID': '0', 'PMIX_BFROP_BUFFER_TYPE': 'PMIX_BFROP_BUFFER_NON_DESC', 'PYTORCH_VERSION': '1.13.0a0+d0d6b1f', 'PMIX_SECURITY_MODE': 'none', 'JUPYTER_PORT': '8888', 'CUDA_DRIVER_VERSION': '520.61.03', '_CUDA_COMPAT_STATUS': 'CUDA Driver OK', 'GITLAB_TOKEN': 'TBD', 'RUNTESTS_LOGS': '/lustre/fsw/adlr/adlr-nlp/mmardani//runtests_logs', 'LOCAL_RANK': '0', 'NVIDIA_PRODUCT_NAME': 'PyTorch', 'LANG': 'en_US.UTF-8', 'PMIX_SERVER_URI21': 'pmix-server.971726;tcp4://127.0.0.1:36747', 'SLURM_STEP_NUM_NODES': '1', 'NPP_VERSION': '11.8.0.86', 'SLURM_CPU_BIND_VERBOSE': 'quiet', 'TENSORBOARD_PORT': '6006', 'PMIX_PTL_MODULE': 'tcp', 'SLURM_PTY_WIN_ROW': '23', 'LS_COLORS': 'rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:', 'SUBMIT_SCRIPTS': '/lustre/fsw/adlr/adlr-others/gpeled/adlr-utils/release/cluster-interface/latest', 'SLURM_JOB_UID': '37774', 'CUFFT_VERSION': '10.9.0.58', 'SLURM_CLUSTER_NAME': 'selene', 'CUDNN_VERSION': '8.6.0.163', 'NSIGHT_COMPUTE_VERSION': '2022.3.0.22', 'DALI_VERSION': '1.17.0', 'SHELL': '/bin/bash', 'CUDNN_VIRTUAL': '', 'SLURM_UMASK': '0027', 'OPENMPI_VERSION': '4.1.4', 'SLURM_STEP_TASKS_PER_NODE': '1', 'TRTOSS_VERSION': '', 'SLURM_PMIXP_ABORT_AGENT_PORT': '44641', 'OMPI_MCA_coll_hcoll_enable': '0', 'SLURM_LOCALID': '0', 'MELLANOX_VISIBLE_DEVICES': '0,1,2,3,6,7,8,9', 'SHARE_OUTPUT': '/lustre/fsw/nvresearch/mmardani/output', 'LESSCLOSE': '/usr/bin/lesspipe %s %s', 'CUSPARSE_VERSION': '11.7.5.86', 'SLURM_JOB_PARTITION': 'interactive', 'SLURM_LAUNCH_NODE_IPADDR': '10.248.1.195', 'SLURM_TASK_PID': '971739', 'OMPI_MCA_btl_tcp_if_include': 'enp97s0f1', 'SLURM_NTASKS': '1', 'BASH_ENV': '/etc/bash.bashrc', 'PMIX_SERVER_TMPDIR': '/var/spool/slurm/d/pmix.3355847.0/', 'PMIX_MCA_ptl': '^usock', 'PWD': '/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr_fulldata/holistic-falcon_2023.01.29_13.49/code', 'SLURM_TOPOLOGY_ADDR': 'selene.quad2.lg09.luna-0197', 'SLURM_NPROCS': '1', 'ADLR_UTILS': '/lustre/fsw/adlr/adlr-others/gpeled/adlr-utils/release/cluster-interface/latest', 'LC_ALL': 'C.UTF-8', 'CUDA_HOME': '/usr/local/cuda', 'SSH_CONNECTION': '10.110.38.243 53432 10.248.1.195 22', 'NVM_CD_FLAGS': '', 'XDG_DATA_DIRS': '/usr/local/share:/usr/share:/var/lib/snapd/desktop', 'IMAGENET_21K': '/lustre/fsw/adlr/adlr-nlp/vkorthikanti/data/imagenet', 'USE_EXPERIMENTAL_CUDNN_V8_API': '1', '_CUDA_COMPAT_PATH': '/usr/local/cuda/compat', 'PYTHONPATH': '/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr_fulldata/holistic-falcon_2023.01.29_13.49/code', 'NVIDIA_VISIBLE_DEVICES': 'all', 'SHARE_SOURCE': '/lustre/fsw/nvresearch/mmardani/source', 'NCCL_VERSION': '2.15.1', 'ADLR_PYTHON': '/lustre/fsw/adlr/adlr-others/gpeled/python/conda3.2020.11/bin/python3', 'SLURM_STEP_NUM_TASKS': '1', 'OPENUCX_VERSION': '1.14.0', 'SUBMIT_ACCOUNT': 'devtech', 'NCCL_VIRTUAL': '', 'HPCX_VERSION': '2.12.1a0', 'SLURM_SRUN_COMM_HOST': '10.248.1.195', 'SHARP_COLL_ENABLE_CUDA': '0', 'IMAGENET': '/lustre/fsw/adlr/adlr-nlp/vkorthikanti/ImageNet_s480_q95', 'PMIX_HOSTNAME': 'luna-0197', 'SLURM_SUBMIT_DIR': '/home/mmardani', 'SLURM_PTY_WIN_COL': '48', 'SLURM_STEP_ID': '0', 'SLURM_NODEID': '0', 'OMP_NUM_THREADS': '1', 'GROUP_RANK': '0', 'ROLE_RANK': '0', 'ROLE_NAME': 'default', 'LOCAL_WORLD_SIZE': '8', 'GROUP_WORLD_SIZE': '1', 'ROLE_WORLD_SIZE': '8', 'TORCHELASTIC_RESTART_COUNT': '0', 'TORCHELASTIC_MAX_RESTARTS': '3', 'TORCHELASTIC_RUN_ID': 'none', 'TORCHELASTIC_USE_AGENT_STORE': 'False', 'NCCL_ASYNC_ERROR_HANDLING': '1', 'TORCHELASTIC_ERROR_FILE': '/tmp/torchelastic_lf4uz4ri/none_vkbbzmyz/attempt_0/0/error.json'})
# get_world_size() 8
# get_rank() 0