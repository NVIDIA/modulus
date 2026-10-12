# SMART for external aerodynamics

This example provides an independent implementation of the model described in
[SMART](https://arxiv.org/abs/2601.18707v2), with separate surface and volume
training recipes for DrivAerML. The model is experimental. It reuses PhysicsNeMo
layers, data utilities, distributed initialization, and checkpointing. It does
not require the authors' code or weights. See
[implementation decisions and provenance](IMPLEMENTATION.md) for differences
from the paper; published SMART or GeoTransolver accuracy is not claimed.

The DrivAerML recipes use **CFD mesh geometry only**. Surface normals and volume
unsigned distance/direction are optional decoder-query inputs. They are added
to the coordinate positional embedding through a linear projection. Geometry
encoding and initial latents use coordinates only.

## Setup and smoke test

Install the checkout following the repository's CUDA installation instructions,
then install the example dependencies from the repository root:

```bash
pip install -e .
cd examples/cfd/external_aerodynamics/smart
pip install -r requirements.txt
```

The smoke test generates original analytic fields, not CFD benchmark data. It
runs on CPU or CUDA and verifies training, checkpoint resume, and inference:

```bash
python make_synthetic_data.py --output ./data
python train.py --config-name=smoke
python train.py --config-name=smoke training.resume=true training.epochs=5
python infer.py --run-dir ./runs/smoke --data-dir ./data/val \
    --output-dir ./predictions/smoke --device cpu --chunk-size 31
```

## Prepare DrivAerML

Obtain the dataset separately and prepare native PhysicsNeMo domain meshes.
The input directory must contain `run_*/domain_*.pdmsh` with a `vehicle` surface
boundary and an interior volume. Surface cells require `pMeanTrim` and
`wallShearStressMeanTrim`; volume points require `UMeanTrim`, `pMeanTrim`, and
`nutMeanTrim`. No CAD files are needed.

Provide disjoint training and validation directories containing the cases for
each split. The preparation command processes every case in its input directory;
it does not decide which cases belong to a split. Prepare complete caches:

```bash
python prepare_drivaerml.py --input /data/native/train \
    --output ./data/drivaerml/train
python prepare_drivaerml.py --input /data/native/val \
    --output ./data/drivaerml/val
```

The cache preserves every surface-cell centroid and volume point, with a shared
permutation for each field's coordinates and targets. It does not impose an
outer geometry cap. Preparation stores nondimensional fields using density
1.205 kg/m³ and speed 30 m/s by default; use `--density` and `--speed` to change
those references. Pressure and shear use density times speed squared, velocity
uses speed, and turbulent viscosity uses speed times vehicle bounding-box length.
Preparation needs enough host memory for a complete case and its temporary
arrays. `--device cuda` can accelerate preparation within a GPU allocation.

Feature recipes also require the native CFD triangles. Set
`data.cfd_feature_mesh_root` to a directory containing `run_*/domain_*.pdmsh` for
all training and validation cases. The root supplies only geometry for query
features. Train/validation membership still comes from the separate cache paths.

## Train separate surface and volume models

The two feature presets use the following inputs and outputs:

| Preset | Decoder-query features | Predicted fields |
| --- | --- | --- |
| `drivaerml_surface_features` | Unit surface normal | Pressure, shear x/y/z |
| `drivaerml_volume_features` | Unsigned distance in metres, closest-surface-to-query unit direction | Velocity x/y/z, pressure, turbulent viscosity |

Surface features are obtained from the nearest native CFD triangle. Volume
features use nearest-triangle distance in physical coordinates; **they are not a
signed SDF**. A retained PhysicsNeMo/Warp BVH computes features for selected
queries. The coordinate-only alternatives are `drivaerml_surface_coordinates`
and `drivaerml_volume_coordinates`.

Launch each model into its own output directory. These examples use four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --config-name=drivaerml_surface_features \
    data.train_path=/data/cache/train data.val_path=/data/cache/val \
    data.cfd_feature_mesh_root=/data/native \
    output_dir=./runs/surface

torchrun --standalone --nproc_per_node=4 train.py \
    --config-name=drivaerml_volume_features \
    data.train_path=/data/cache/train data.val_path=/data/cache/val \
    data.cfd_feature_mesh_root=/data/native \
    output_dir=./runs/volume
```

Both recipes use six layers, width 448, eight heads, 2,048 latents, and **16,384
geometry samples per encoder layer**. Latents and each layer's geometry sample
are drawn from the complete CFD vehicle surface-centroid cloud. Sampling is
without replacement within a draw, and different layers may overlap. Geometry
samples change with the case and epoch during training. Private geometry RNG
streams preserve field sampling when the geometry budget changes. Evaluation
uses a fixed geometry seed. Sampling happens before embedding; the model does
not embed or attend to the complete cloud simultaneously. The raw geometry pool
and native BVH still consume memory proportional to mesh size.

Each enabled field uses 200,000 training queries. Query coordinates and geometry
samples are separate inputs, although surface queries share the CFD boundary.
Coordinates are centered on the CFD surface-centroid bounding box and scaled by
`[12.0, 4.5, 3.25]`. Conditions are density and speed. The shared configuration
contains fixed field standardization statistics from the existing DrivAerML
recipe; review or replace them when changing the dataset. Preserve the same
normalization during validation and inference.

Training uses FP32, standardized MSE, and PhysicsNeMo's combined Muon/AdamW
optimizer at learning rate 0.001. The learning rate halves every 10,900 successful
updates. The endpoint is 54,500 successful updates, equivalent to 500 epochs for
436 training cases and four ranks with one case per rank. Different split sizes
or GPU counts change the epoch/update relationship and effective batch size.
Set `training.max_updates=null` to use the epoch endpoint instead. Variable-sized
CFD meshes require batch size one per rank. Q/K normalization and a bounded
learned attention temperature are explicit extensions of the paper.

## Validation, inference, and resume

DrivAerML training validation evaluates 200,000 fixed queries per enabled field
and case. Metrics are reduced across query chunks before computing per-case
errors; distributed validation does not duplicate cases. `best.mdlus` is selected
by the validation sum of channel-averaged relative L2 errors, not a pressure-only
or force metric. Use a predefined checkpoint/endpoint for comparisons and report
which fields and point sets were evaluated.

Inference evaluates **all points in the supplied cache** using saved
preprocessing and feature definitions:

```bash
python infer.py --run-dir ./runs/surface --data-dir /data/cache/val \
    --output-dir ./predictions/surface --device cuda --chunk-size 32768
python infer.py --run-dir ./runs/volume --data-dir /data/cache/val \
    --output-dir ./predictions/volume --device cuda --chunk-size 32768
```

Output directories must be empty. Predictions are written to memory-mapped
`.npy` files, accompanied by case and checkpoint manifests. DrivAerML outputs
are de-standardized nondimensional fields in **cache order**, which differs
from native mesh order. This command does not export native VTK files or compute
drag/lift. Recover physical units using the same reference factors as preparation
before downstream integration. Full query arrays occupy host memory, while
chunking bounds decoder activation memory. Changing the query chunk size does
not change the model's geometry context.

Runs save the resolved configuration, `metrics.jsonl`, model files, and paired
optimizer/scheduler/scaler checkpoints. Resume the same run with the same
configuration, dataset, and world size:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
    --config-name=drivaerml_surface_features \
    data.train_path=/data/cache/train data.val_path=/data/cache/val \
    data.cfd_feature_mesh_root=/data/native \
    output_dir=./runs/surface training.resume=true
```

Resume restores the latest complete epoch checkpoint. There is no mid-epoch
resume. Source data must remain unchanged; configuration checks do not hash the
complete dataset. Avoid concurrent writers to a run directory. Model-only
inference files such as `best.mdlus` do not contain optimizer state.

## Generic point-cloud input

The default `config.yaml` also supports joint training on one `.npz` file per
case. This is a configurable paper-inspired example, not an exact benchmark
reproduction. The synthetic smoke test uses this format:

| Array | Shape | Meaning |
| --- | --- | --- |
| `geometry` | `(N, 3)` | Geometry coordinates, separate from field queries |
| `surface_queries` | `(Ms, 3)` | Surface coordinates |
| `surface_targets` | `(Ms, Cs)` | Aligned surface targets |
| `volume_queries` | `(Mv, 3)` | Volume coordinates |
| `volume_targets` | `(Mv, Cv)` | Aligned volume targets |
| `parameters` | `(P,)` | Required when `parameter_channels > 0` |

Arrays must contain finite floating-point values. Disable a head by setting its
channel count to zero and omit its arrays. Targets are unnecessary for inference.
Set coordinate and condition normalization explicitly; generic targets retain
the supplied units. The generic objective sums channel-averaged relative L2
errors and uses Lion. Use `training.precision=float32` on CPU. Generic validation
evaluates every supplied query.
