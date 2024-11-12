import os


def adjust_cluster_paths(params):

    if 'SLURM_CLUSTER_NAME' in os.environ:

        if os.environ['SLURM_CLUSTER_NAME'] == 'eos':

            params['location'] = params['eos_location']
            params['exp_dir'] = params['eos_exp_dir']
            params['regression_model_basepath'] = params['eos_regression_model_basepath']


    return(params)
