# Implementation decisions and provenance

## Source boundary

This implementation was written from
[SMART, arXiv:2601.18707v2](https://arxiv.org/html/2601.18707v2), including its
equations, Figure 2, and appendices, together with NVIDIA PhysicsNeMo components.
No research model source, weights, preprocessing code, or numerical fixtures
were used. Synthetic test data and regression fixtures were generated from this
implementation.

An earlier licensing discussion viewed the research repository's README and
license, including README usage snippets. Those snippets were not used as an
implementation specification or copied. This is documented independent authorship
from the paper, not a claim of zero prior exposure to repository material.

The implementation is experimental and does not claim weight compatibility,
exact parameter-count agreement, or reproduction of the paper's accuracy.

## Paper mapping and engineering choices

| Paper reference | Implementation |
| --- | --- |
| Section 3.1.1 | Learned scale/shift inside sinusoidal coordinate phases |
| Section 3.1.2, equations 2–4 | Geometry-derived latent points and independently resampled geometry attention at each layer |
| Equation 5 | Conditions modulate activated hidden MLP features |
| Section 3.1.3, equation 7 | Each decoder layer attends to its matching intermediate geometry context |
| Section 3.1.3 / Table 13 | Optional encoder/decoder refinement sharing, enabled in the presets |
| Section 3.2 | Random training queries and independent, chunked inference queries |
| Appendix I.2 | Generic recipe sums surface and volume channel-averaged relative L2 errors |

The paper does not fully specify all attention heads, intermediate widths,
normalization/residual placement, or positional axis-width allocation. This
implementation makes those choices explicit:

- Eight attention heads, GELU, MLP expansion four, pre-layer normalization,
  residual attention/MLP updates, and no dropout by default.
- Allocate `2 * ceil(hidden_dim / 6)` sinusoidal features per coordinate axis,
  concatenate all three axes, and project to the hidden width. A shared scalar
  coordinate MLP modulates each axis's phases, initialized to identity. Phase
  arithmetic uses FP32 under autocast.
- Geometry and query coordinates share the positional embedding. Conditions
  modulate hidden activations as `(1 + delta_scale) * h + shift`, initialized to
  identity. Query features, when enabled, are added through zero-initialized
  linear projections; they do not replace the coordinate positional embedding.
- Decoder layers consume intermediate geometry-attention contexts. The final
  encoder refinement is omitted because no prediction consumes its output.
- Uniform geometry sampling is without replacement and precedes embedding. If a
  cloud is smaller than the requested count, use all its points. Evaluation
  uses a private seeded generator. The CFD recipe supplies independent training
  geometry streams, separate from field-query sampling.
- Relative errors floor squared target norms at `1e-8`. Relative-L2 training
  floors the ratio before square root to keep gradients finite at zero error.

Parameter-count agreement is not established by these choices. Tests of query
independence, chunked inference, conditioning, sampling, and serialization verify
this implementation's contracts rather than equivalence to the authors' code.

## DrivAerML adaptation

The CFD recipes retain the earlier additive positional/query-feature model.
They train separate surface and volume models with six layers, width 448, 2,048
latents, 16,384 geometry samples per layer, and 200,000 field queries. Surface
queries receive unit normals. Volume queries receive unsigned nearest-triangle
distance in metres and closest-surface-to-query direction in physical coordinates.
Geometry encoding receives coordinates only. Signed, scaled-coordinate SDFs and
joint coordinate/feature MLP embeddings are not included.

These recipes deliberately differ from the paper's datasets and training
settings. They use standardized MSE, Muon/AdamW, an update-based step schedule,
and FP32. They add optional per-head Q/K RMS normalization and a learned
attention temperature `max_temperature * sigmoid(logit)`, initialized to one
and bounded by four in the CFD presets. Neither is attributed to the paper.
The generic configuration leaves these extensions disabled.

The geometry pool comprises the complete CFD vehicle surface-centroid cloud;
it is not a CAD cloud and does not establish mesh-independent generalization.
Initial latents and per-layer geometry draws use this pool. There is no
intermediate 300,000-point cap. Field queries remain separate from geometry
inputs, and sampled geometry is embedded before attention. The CFD triangle
mesh is retained for geometric feature queries, without dense learned embeddings.

Several training settings and standardization statistics follow the existing
PhysicsNeMo external-aerodynamics transformer example. This does not match all
GeoTransolver inputs: its geometric source, signed/scaled distance definition,
geometry embedding, and learned multiscale context differ. Equal sample counts
therefore do not imply equal geometric information. Report sampled versus
full-field evaluation explicitly, and use held-out data for final accuracy claims.

## PhysicsNeMo reuse and contribution scope

The model inherits `physicsnemo.Module` and uses `ModelMetaData`, `Mlp`, and
`FCLayer`. The example reuses PhysicsNeMo readers, dataset transforms, data
loading, relative-error reductions, distributed initialization, checkpoint
utilities, Muon/AdamW, and Warp nearest-triangle queries. PyTorch scaled
dot-product attention supplies cross-attention with independent queries.

Following **MOD-001**, models “must inherit from `physicsnemo.Module`.” Following
**MOD-002a**, new model classes belong in `physicsnemo/experimental/models`.
These keep the model's experimental status explicit. The model adds no core
package dependencies. Following **EXT-001**, “Example applications may list
additional packages under `examples/**/requirements.txt`”; Lion and TensorStore
remain example dependencies.

The contribution includes field training and chunked field inference. It excludes
cluster launchers, experiment histories, pretrained weights, force-supervised
training, and native force-evaluation tooling. Public recipes contain portable
paths that users override for their own dataset layout.
