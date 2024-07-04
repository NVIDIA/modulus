from typing import Dict
from azure.ai.ml import command, Input, Output, MLClient,MpiDistribution,PyTorchDistribution
from azure.ai.ml.entities import UserIdentityConfiguration
from azure.identity import DefaultAzureCredential

from examples.cfd.aero_graph_net.settings import Settings

def build_tags(tags:Dict) -> Dict[str, str]:
    """Builds the tags for the job."""
    return {**tags, **dict({
        "experiment_name": "ahmed_body_v2",
        "experiment_type": "training",
    })}

def run_training_job():
    settings=Settings()

    ml_client = MLClient(credential=DefaultAzureCredential(), subscription_id= settings.subscription_id, resource_group_name=settings.resource_group, workspace_name= settings.workspace)
    tags=build_tags(settings.tags)

    built_inputs=create_inputs(settings)
    # define the command
    command_job = command(
        code=".",
        command="python examples/cfd/aero_graph_net/train.py +experiment=ahmed/mgn output=${{outputs.checkpoint_dir}} data.data_dir=${{inputs.data_dir}}",
        environment=settings.aml_environment,
        display_name= settings.display_name,
        experiment_name="ahmed_body_v2",
        shm_size="5g",
        identity= UserIdentityConfiguration(),
        tags=tags,
        inputs=built_inputs,
        environment_variables={
            # "DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED": False, # Enable block-based caching
            # "DATASET_MOUNT_BLOCK_FILE_CACHE_ENABLED": False, # Disable caching on disk
            # "DATASET_MOUNT_MEMORY_CACHE_SIZE": 0, # Disabling in-memory caching
            "DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED": False, # disable block-based caching
            "HYDRA_FULL_ERROR": 1,

            },

        outputs={
            "output_dir" : Output(
                mode="rw_mount",
                path=settings.output_dir,
                type="uri_folder",
                name="ahmed_output",
            ),         
            "checkpoint_dir" : Output(
                mode = "rw_mount",
                path= settings.ckpt_dir,
                type="uri_folder",
                name="ahmed_output",
            )
        },

        compute=settings.compute_name,
    )
    # set openmpi parrallel mode
    if settings.parallel_mode=="openmpi":
        command_job.environment_variables["OMPI_ALLOW_RUN_AS_ROOT"]=1
        command_job.environment_variables["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"]=1
        command_job.environment_variables["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"]="OPENMPI"
        command_job.distribution=MpiDistribution(process_count_per_instance=4)
    elif settings.parallel_mode=="pytorch":
        command_job.distribution=PyTorchDistribution(process_count_per_instance=4)
        command_job.environment_variables["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"]="ENV"
    # submit the command
    returned_job = ml_client.jobs.create_or_update(command_job)
    
    # get a URL for the status of the job
    returned_job.studio_url

def create_inputs(settings):
    inputs = dict()
    inputs["data_dir"] = Input(
                type="uri_folder",
                path=settings.data_dir,
                mode = "download"
            )
    inputs["epochs"] =Input(type="integer", default=(settings.hydra_epochs or 100))
    inputs["checkpoint_save_freq"]= Input(type="integer", default=(settings.hydra_checkpoint_save_freq or 5))
    return inputs

if __name__ == "__main__":
    run_training_job()