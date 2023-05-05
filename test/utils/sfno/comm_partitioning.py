num_ranks = 32

# we want this partitioning
model_parallel_size = 4
model_parallel_stride = 1
num_model_groups = num_ranks // model_parallel_size
num_stride_groups = num_model_groups // model_parallel_stride

model_groups = []
for i in range(num_stride_groups):
    for j in range(model_parallel_stride):
        start = j+ i * (model_parallel_size*model_parallel_stride)
        end = start + model_parallel_size * model_parallel_stride
        model_groups.append(range(start, end, model_parallel_stride))


print(model_groups)
print([sorted(list(i)) for i in zip(*model_groups)])
