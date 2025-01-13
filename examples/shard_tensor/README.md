# TODO - UPDATE EVERYTHING :)


THESE EXAMPLES ARE A HEAVY WORK IN PROGRESS.  If you're here, you're looking at early versions 
of code that is still in a prototype stage, with not many tests and shard edges.  That said,
I would welcome feedback on all of this - as much as you'd like to give!

## DTensor:
Check out some of the dtensor examples to see simple examples of creating and using dtensors
(and why they fail with irregular data).

## ShardTensor:
Check out the shard tensor examples to see ~roughly the same examples but now functioning across 
uneven sharded tensors.

## Examples: 

### Convolutions
The convolution examples are a bit outdated but I will update them.

The Halo interface backward pass is not implemented correctly yet, be careful!

### Neighborhood Attention

There is a straightforward implemenation of a neighborhood attention mechanism which is 
easy enough to understand.  The more complicated technique of correctly dispatching
operations ShardTensors and capturing the appropriate output placements is still WIP.

