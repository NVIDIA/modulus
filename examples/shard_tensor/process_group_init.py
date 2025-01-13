import socket

from modulus.distributed import DistributedManager
# from modulus.distributed import ProcessGroupNode, ProcessGroupConfig


    
def spawn_groups(data_parallel_size=2, sub_parallel_size=2):
    
    # The distributed manager will handle all of the communicator sharding, as necessary:
    dm = DistributedManager()
    
    dm.initialize_mesh((data_parallel_size,sub_parallel_size), ("ddp", "domain"))
    



if __name__ == "__main__":

    # This will trigger the global initialization across all processes:
    DistributedManager.initialize()
    
    dm = DistributedManager()

    spawn_groups(3, 2)

    world_size  = dm.world_size
    global_rank = dm.rank
    local_rank  = dm.local_rank
    
    placement_str = f"Rank {global_rank} of {world_size} has local rank {local_rank} on {socket.gethostname()}, and group placement: \n"
    
    for group_name in dm.group_names:
        group_str = f"\tIn group {group_name}, this is rank {dm.group_rank(group_name)} of {dm.group_size(group_name)} (full placement: {dm._group_ranks[group_name]}) \n"
        placement_str += group_str
        
    print(placement_str)
    DistributedManager.cleanup()

    # input_data = torch.rand((1, 16, 256, 256 ))


    # Set up a config group that is 4x2, 4 data parallel and 2 sub parallel
