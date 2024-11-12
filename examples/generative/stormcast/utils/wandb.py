import os
import pwd

# Perlmutter uids for different user groups
nersc_lbl = ['pharring', 'shas1693', 'amahesh', 'shaoming', 'jwillard']
nersc_nvidia = ['jpathak', 'nbren12', 'yacohen']

# Entity, project for different wandb spaces
wandb_dict = {
              'lbnl': ['weatherbenching', 'hrrr'],
              'nvidia': ['nv-research-climate', 'hrrr'],
             }

def get_wandb_names():
    """
    Query local env for uid and machine name, return corresponding wandb entites
    """
    if 'NERSC_HOST' in os.environ:
        # Perlmutter uids apply
        uid = pwd.getpwuid(os.getuid())[0]
        if uid in nersc_lbl:
            return wandb_dict['lbnl']
        else:
            return wandb_dict['nvidia']
    else:

        if 'SLURM_CLUSTER_NAME' in os.environ:

            if os.environ['SLURM_CLUSTER_NAME'] == 'eos':

                return wandb_dict['nvidia']


