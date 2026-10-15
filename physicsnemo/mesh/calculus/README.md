# Discrete Calculus on Simplicial Meshes

## Overview

This module implements differential operators (gradient, divergence, curl,
Laplacian) and integration for simplicial meshes. It uses two complementary
approaches:

1. Discrete Exterior Calculus (DEC): a differential-geometry framework based
   on Desbrun et al. (2005) and Hirani (2003). Fields live on vertices, edges,
   and dual cells, and the operators follow the discrete Stokes theorem.
2. Weighted least squares (LSQ): the standard CFD approach. Each vertex or
   cell fits a local linear model to the values at its neighbors.

For a guide to choosing between them, see
[Choosing Between DEC and LSQ](#choosing-between-dec-and-lsq). For the API
reference, see the
[calculus page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/calculus.html).

### Setup

The examples on this page run in order. This block defines the mesh and the
fields that the later blocks use: a sphere (a 2D surface in 3D) with a scalar
field and a vector field at its vertices.

```python
import torch
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

mesh = sphere_icosahedral.load(subdivisions=3).to(torch.float64)
x, y, z = mesh.points.unbind(dim=-1)

scalar_field = x * y + z  # (n_points,)
vector_field = torch.stack([-y, x, z], dim=-1)  # (n_points, 3)
cell_values = mesh.cell_centroids[:, 0]  # (n_cells,)

mesh.point_data["pressure"] = scalar_field
mesh.point_data["temperature"] = z**2
```

---

## Discrete Exterior Calculus (DEC)

DEC builds discrete operators that satisfy discrete versions of continuous
theorems. The exterior derivative satisfies the discrete Stokes theorem
exactly, and the angle-defect Gaussian curvature satisfies Gauss-Bonnet
exactly on closed surfaces.

### Core DEC Operators

#### Laplace-Beltrami Operator

```python
from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

# Intrinsic Laplacian:
#   Lap_f(v) = (1 / |star v|) sum (|star e| / |e|) * (f_neighbor - f_v)
laplacian = compute_laplacian_points_dec(mesh, scalar_field)
```

The sign convention is the analyst's: the Laplacian of $x^2$ on a line is
$+2$, and the operator is negative semidefinite.

Properties:

- Uses cotangent weights derived from the FEM stiffness matrix (see below).
- In 2D: equivalent to $\tfrac{1}{2}(\cot \alpha + \cot \beta)$[^meyer2003-eq5].
- In 1D: gives $1/|e|$ (standard finite-difference second derivative).
- In 3D and higher: gives the dihedral-angle cotangent weights, computed from
  the inverse Gram matrix.
- Normalized by the dual 0-cell volumes $|\star v|$: circumcentric (Voronoi)
  volumes in 1D and 2D, a barycentric approximation in 3D and higher (see
  [Circumcentric Dual Volumes](#circumcentric-dual-volumes-voronoi-cells)).
- Exact for linear functions at interior vertices of flat meshes
  (codimension 0, where the manifold dimension equals the spatial dimension).
  Boundary vertices have an incomplete 1-ring and give large errors.
- Works on manifolds of any dimension embedded in any ambient space.

Cotangent weights via the FEM stiffness matrix (n-dimensional):

For an $n$-simplex with vertices $v_0, \ldots, v_n$, the cotangent weight
for edge $(i, j)$ is

$$
w_{ij} = -|\sigma| \, (\nabla \lambda_i \cdot \nabla \lambda_j),
$$

where $\lambda_i$ are barycentric coordinate functions and $|\sigma|$ is
the cell volume. The gradient dot products are computed from the inverse
Gram matrix:

$$
\begin{aligned}
E &= [v_1 - v_0, \ldots, v_n - v_0]
   \quad (n \times d \text{ edge matrix}) \\
G &= E E^\top
   \quad (n \times n \text{ Gram matrix}) \\
\nabla \lambda_k \cdot \nabla \lambda_l &= (G^{-1})_{k-1,\, l-1}
   \quad \text{for } k, l \geq 1.
\end{aligned}
$$

This generalizes the classical 2D cotangent formula to arbitrary
dimensions. See [^hirani2003-eq642] and [^meyer2003-eq8].

#### Exterior Derivative

```python
from physicsnemo.mesh.calculus._exterior_derivative import (
    exterior_derivative_0,
    exterior_derivative_1,
)

# d: Omega^0 -> Omega^1 (0-forms to 1-forms)
#   df([vi, vj]) = f(vj) - f(vi)
edge_1form, edges = exterior_derivative_0(mesh, scalar_field)

# d: Omega^1 -> Omega^2 (1-forms to 2-forms; circulation around faces)
# Requires n_manifold_dims >= 2.
face_2form, faces = exterior_derivative_1(mesh, edge_1form, edges)
```

Properties:

- $d \circ d = 0$ (exact by construction).
- Discrete Stokes theorem:
  $\langle d\alpha, c\rangle = \langle \alpha, \partial c\rangle$
  (true by definition).

See [^desbrun2005-s5] and [^hirani2003-ch3].

#### Hodge Star

```python
from physicsnemo.mesh.calculus._hodge_star import hodge_star_0, hodge_star_1

# star: Omega^0 -> Omega^n (vertex values to dual n-cells)
#   star_f(star v) = f(v) * |star v|
star_f = hodge_star_0(mesh, scalar_field)

# star: Omega^1 -> Omega^(n-1) (edge values to dual (n-1)-cells)
star_df = hodge_star_1(mesh, edge_1form, edges)
```

Properties:

- Preserves averages between primal and dual cells (see equation below).
- $\star\star\alpha = (-1)^{k(n-k)} \alpha$.
- Uses circumcentric (Voronoi) dual cells, not barycentric ones, in 1D and
  2D.

$$
\frac{\langle \alpha, \sigma\rangle}{|\sigma|}
  = \frac{\langle \star\alpha, \star\sigma\rangle}{|\star\sigma|}
$$

See [^hirani2003-def411] and [^desbrun2005-s6].

#### Sharp and Flat Operators

```python
from physicsnemo.mesh.calculus._sharp_flat import sharp, flat

# sharp: Omega^1 -> X (1-forms to vector fields)
grad_vector = sharp(mesh, edge_1form, edges)  # (n_points, n_spatial_dims)

# flat: X -> Omega^1 (vector fields to 1-forms)
one_form = flat(mesh, vector_field, edges)  # (n_edges,)
```

Implementation:

- Sharp ($\sharp$): equation 5.8.1 with support volume intersections and
  barycentric gradients[^hirani2003-eq581].
- Flat ($\flat$): PDP-flat with averaged endpoint vectors[^hirani2003-s56].

Sharp and flat are not exact inverses in discrete
DEC[^hirani2003-prop553]. This is a property of the discrete theory, not a
bug.

See [^hirani2003-ch5].

### Gradient via DEC

```python
from physicsnemo.mesh.calculus.gradient import compute_gradient_points_dec

# Computes: grad(f) = sharp(df)
grad_f = compute_gradient_points_dec(mesh, scalar_field)  # (n_points, 3)
```

This composes the exterior derivative and the sharp operator. It is exact for
linear functions on flat meshes (codimension 0), including at boundary
vertices.

### Divergence via DEC

```python
from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec

div_v = compute_divergence_points_dec(mesh, vector_field)  # (n_points,)
```

This computes $\star_0^{-1}\, d^{\ast}\, \star_1\, \flat(X)$: the net flux
through the boundary of each dual cell, per unit dual volume. It is exact for
linear vector fields at interior vertices of uniform grids and of 1D meshes.
On irregular triangle or tetrahedral meshes it is not exact, and its pointwise
error does not shrink as the mesh refines.

---

## Weighted Least-Squares (LSQ) Methods

LSQ methods fit a local linear model at each vertex or cell. They work on
irregular meshes and on tensor-valued fields.

### Gradient

```python
from physicsnemo.mesh.calculus.gradient import (
    compute_gradient_points_lsq,
    compute_gradient_cells_lsq,
)

# At vertices
grad = compute_gradient_points_lsq(
    mesh,
    scalar_field,
    weight_power=2.0,  # Inverse-distance weighting exponent (the default)
    intrinsic=False,  # Set True for tangent-space gradients on manifolds
)

# At cell centers
grad_cells = compute_gradient_cells_lsq(mesh, cell_values)  # (n_cells, 3)
```

Properties:

- Exact for constant and linear fields on flat meshes (codimension 0),
  including at boundary vertices.
- First-order accurate, $O(h)$, for smooth fields.
- Supports intrinsic (tangent-space) gradients for curves in 2D or 3D and
  surfaces in 3D.
- Works for scalar and tensor fields: a field of shape `(n_points, ...)` gives
  a gradient of shape `(n_points, n_spatial_dims, ...)`.
- Requires `n_spatial_dims <= 3`.

### Divergence

```python
from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

div_v = compute_divergence_points_lsq(mesh, vector_field)  # (n_points,)
```

Computes the divergence

$$
\operatorname{div}(v) = \partial_x v_x + \partial_y v_y + \partial_z v_z
$$

as the trace of the LSQ Jacobian. `compute_divergence_cells_lsq` does the same
at cell centers.

### Curl (3D Only)

```python
from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

curl_v = compute_curl_points_lsq(mesh, vector_field)  # Requires n_spatial_dims = 3
```

Computes the curl from the antisymmetric part of the LSQ Jacobian.
`compute_curl_cells_lsq` does the same at cell centers.

---

## Circumcentric Dual Volumes (Voronoi Cells)

### Implementation

```python
from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0

dual_vols = compute_dual_volumes_0(mesh)  # |star v| for each vertex
```

Algorithm (dimension-specific):

1D manifolds (edges):

- Each vertex gets half the length of each incident edge.
- Exact for piecewise linear 1-manifolds.

2D manifolds (triangles):

- Acute triangles: circumcentric Voronoi formula[^meyer2003-eq7],

$$
|\star v| = \tfrac{1}{8} \sum_{e \ni v} \|e\|^2 \cot \theta_{\mathrm{opp}}(e).
$$

- Obtuse triangles: mixed-area subdivision[^meyer2003-fig4],

$$
|\star v| = \begin{cases}
  \mathrm{area}(T) / 2 & \text{if obtuse at } v, \\
  \mathrm{area}(T) / 4 & \text{otherwise}.
\end{cases}
$$

3D and higher manifolds (tetrahedra, etc.):

- Barycentric approximation: $|\star v| = \sum_{\sigma \ni v} |\sigma| / (n + 1)$.
- A rigorous circumcentric dual requires "well-centered"
  meshes[^desbrun2005].

In every dimension, the dual volumes tile the mesh:
$\sum_v |\star v| = |M|$ up to floating-point rounding.

See [^meyer2003-s32_34], [^desbrun2005-s3], and [^hirani2003-def245].

---

### Known Behavior (Not Bugs)

$\operatorname{div}(\nabla f)$ is close to $\Delta f$, but not equal to it:

- In discrete DEC, sharp ($\sharp$) and flat ($\flat$) are not exact
  inverses[^hirani2003-prop553].
- Therefore the composed operator differs from
  `compute_laplacian_points_dec`. Away from the boundary of a regular mesh,
  both converge to $\Delta f$ as the mesh refines. On irregular meshes, the
  pointwise error of either one need not go to zero.
- To compute a Laplacian, call `compute_laplacian_points_dec` directly.

3D dual volumes use a barycentric approximation:

- A rigorous circumcentric dual requires "well-centered" meshes[^desbrun2005].
- No mixed-volume formula for obtuse tetrahedra exists in the literature.
- The barycentric approximation is standard practice.

---

## API Reference

The full API reference, with every signature, is on the
[calculus page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/calculus.html).

### High-Level Interface

```python
# Unified interface for gradients of point_data fields
mesh_with_grad = mesh.compute_point_derivatives(
    keys=["pressure", "temperature"],
    method="lsq",  # or "dec"
    gradient_type="extrinsic",  # "intrinsic" (the default), "extrinsic", or "both"
)

# Access results
grad_p = mesh_with_grad.point_data["pressure_gradient"]  # (n_points, n_spatial_dims)
```

`mesh.compute_cell_derivatives` does the same for `cell_data` with LSQ.

### Direct Operator Calls

```python
from physicsnemo.mesh.calculus import (
    compute_gradient_points_lsq,
    compute_gradient_points_dec,
    compute_divergence_points_lsq,
    compute_divergence_points_dec,
    compute_curl_points_lsq,
    compute_laplacian_points_dec,
)

# Gradient (LSQ or DEC)
grad = compute_gradient_points_lsq(
    mesh, scalar_field, weight_power=2.0, intrinsic=False
)
grad = compute_gradient_points_dec(mesh, scalar_field)

# Divergence (LSQ or DEC)
div = compute_divergence_points_lsq(mesh, vector_field)
div = compute_divergence_points_dec(mesh, vector_field)

# Curl (n_spatial_dims = 3 only)
curl = compute_curl_points_lsq(mesh, vector_field)

# Laplacian (DEC)
laplacian = compute_laplacian_points_dec(mesh, scalar_field)
```

### Integration

```python
from physicsnemo.mesh.calculus import integrate

# P1 quadrature of a point field; exact for piecewise-linear fields
total = integrate(mesh, "pressure", data_source="points")
```

`integrate_flux`, `integrate_moment`, and the measure helpers in
`physicsnemo.mesh.calculus.measure` complete the integration API.

---

## Performance

All operations are vectorized (no Python loops over mesh elements):

- Gradient, divergence, curl: `O(n_points * avg_degree)`.
- Laplacian: `O(n_edges)`.
- Dual volumes: `O(n_cells)`, computed once and cached on the mesh.

All operators run on the device of the mesh, CPU or GPU.

---

## Module Structure

```text
physicsnemo/mesh/calculus/
├── __init__.py                    # Public API
├── derivatives.py                 # High-level interface (compute_point_derivatives)
├── gradient.py                    # Gradient (LSQ + DEC)
├── divergence.py                  # Divergence (LSQ + DEC)
├── curl.py                        # Curl (LSQ, 3D only)
├── laplacian.py                   # Laplace-Beltrami (DEC)
├── integration.py                 # Integration of cell and point fields
├── measure.py                     # Cell and point integration measures
│
├── _exterior_derivative.py        # DEC: exterior derivative d
├── _hodge_star.py                 # DEC: Hodge star
├── _sharp_flat.py                 # DEC: sharp and flat operators
│
├── _lsq_reconstruction.py         # LSQ: gradient reconstruction (ambient space)
├── _lsq_intrinsic.py              # LSQ: intrinsic gradients (tangent space)
├── _neighborhoods.py              # LSQ: batched neighborhood iteration
└── _pca_tangent.py                # PCA tangent-space estimation
```

```text
physicsnemo/mesh/geometry/
├── dual_meshes.py                 # Dual 0-cell volumes, circumcenters, cotangent weights
├── support_volumes.py             # Support volume intersections for DEC
└── interpolation.py               # Barycentric function gradients
```

---

## Usage Examples

### Example 1: Laplace-Beltrami on a Curved Surface

```python
from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

# Scalar field on the sphere from the setup block
temperature = mesh.point_data["temperature"]

# Compute the intrinsic Laplacian
laplacian = compute_laplacian_points_dec(mesh, temperature)

# Use for diffusion: dT/dt = kappa * Lap T
mesh.point_data["laplacian_T"] = laplacian
```

### Example 2: Gradient on a Manifold (Intrinsic)

```python
from physicsnemo.mesh.calculus.gradient import compute_gradient_points_lsq

# Compute the gradient in the tangent space of the surface
grad_intrinsic = compute_gradient_points_lsq(
    mesh,
    scalar_field,
    intrinsic=True,  # Solves in tangent space
)

# The result is perpendicular to the vertex normals
assert torch.allclose(
    (grad_intrinsic * mesh.point_normals).sum(dim=-1),
    torch.zeros(mesh.n_points, dtype=mesh.points.dtype),
    atol=1e-6,
)
```

### Example 3: Vector Calculus Identities

The continuous identities $\nabla \times \nabla f = 0$ and
$\nabla \cdot (\nabla \times v) = 0$ hold for LSQ operators only
approximately. They hold to rounding error for linear fields. For quadratic
fields on `cube_volume`, the residuals are at rounding level far from the
boundary, and 0.35 to 0.36 at vertices near it.

```python
from physicsnemo.mesh.calculus import (
    compute_gradient_points_lsq,
    compute_divergence_points_lsq,
    compute_curl_points_lsq,
)
from physicsnemo.mesh.primitives.volumes import cube_volume

volume = cube_volume.load(subdivisions=4).to(torch.float64)
vx, vy, vz = volume.points.unbind(dim=-1)

# curl(grad(f)) for a linear f: zero to rounding error
f_linear = vx + 2 * vy + 3 * vz
curl_grad_f = compute_curl_points_lsq(
    volume, compute_gradient_points_lsq(volume, f_linear)
)
assert curl_grad_f.abs().max() < 1e-10

# div(curl(v)) for a quadratic v: not zero near the boundary
v_quadratic = torch.stack([vy * vz, vx**2, vx * vy], dim=-1)
div_curl_v = compute_divergence_points_lsq(
    volume, compute_curl_points_lsq(volume, v_quadratic)
)
print(f"max |div(curl v)| = {div_curl_v.abs().max():.2f}")  # 0.35
```

---

## Dimension Support

Which operators run, by manifold dimension $n$ (columns) and spatial
dimension $d$. Codimension is $d - n$. Entries list the supported spatial
dimensions; "-" means the call raises an error.

<!-- markdownlint-disable MD013 -->
| Operator | $n = 1$ | $n = 2$ | $n = 3$ | $n \geq 4$ |
| -------- | ------- | ------- | ------- | ---------- |
| Gradient, divergence (LSQ) | $d \leq 3$ | $d \leq 3$ | $d = 3$ | - |
| Gradient (LSQ, `intrinsic=True`) | $d \leq 3$ | $d \leq 3$ | $d = 3$ | - |
| Curl (LSQ) | $d = 3$ | $d = 3$ | $d = 3$ | - |
| Gradient (DEC) | any $d$ | $d \leq 3$ | $d = 3$ | - |
| Divergence (DEC) | any $d$ | any $d$ | any $d$ | any $d$ |
| Laplacian (DEC) | any $d$ | any $d$ | any $d$ | any $d$ |
| Hodge star, dual volumes | any $d$ | any $d$ | any $d$\* | any $d$\* |
| `exterior_derivative_1` | - | any $d$ | any $d$ | any $d$ |
<!-- markdownlint-enable MD013 -->

\*Dual volumes use the barycentric approximation for $n \geq 3$.

The LSQ operators require $d \leq 3$. The DEC gradient needs barycentric
gradients, which exist for segments in any $d$, triangles in $d \leq 3$, and
tetrahedra in $d = 3$. The DEC Laplacian uses the FEM cotangent weights in
every dimension.

---

## Choosing Between DEC and LSQ

Use DEC when:

- You need operators with discrete structure: $d \circ d = 0$ and the
  discrete Stokes theorem.
- You work with differential geometry (curvatures, etc.).
- You compute a Laplacian on a manifold.
- You work in more than three spatial dimensions (DEC divergence and
  Laplacian).

Use LSQ when:

- You need a general-purpose gradient, divergence, or curl.
- You work with irregular or poor-quality meshes.
- You compute derivatives of tensor fields.
- You need values at cell centers (`compute_gradient_cells_lsq` and
  related functions).

Both methods:

- Are first-order accurate, $O(h)$, for gradients of smooth fields.
- Work on irregular meshes.
- Are fully vectorized.
- Run on the GPU.

---

## Limitations and Future Work

### Current Limitations

1. 3D and higher dual volumes use the barycentric approximation (standard
   practice).
   - A rigorous circumcentric dual requires "well-centered" meshes.
   - A mixed volume for obtuse tetrahedra is an open research problem.
   - The Laplacian cotangent weights are exact in all dimensions (FEM
     stiffness matrix). Only the normalization by dual volume is an
     approximation.

2. Sharp and flat are not exact inverses: $\sharp \circ \flat \neq \mathrm{id}$
   in discrete DEC.
   - This is fundamental to the discrete theory[^hirani2003-prop553].
   - It makes $\operatorname{div}(\nabla f)$ differ from $\Delta f$.

3. Boundary effects: the cotangent Laplacian assumes complete 1-ring
   neighborhoods.
   - Boundary vertices give large errors.
   - `mean_curvature_vertices` returns `NaN` at boundary vertices by default
     (`include_boundary=False`).

### Future Enhancements

1. Well-centered mesh detection for rigorous 3D dual volumes.
2. Additional DEC operators: wedge product, interior product, Lie
   derivative.
3. Higher-order LSQ with extended stencils.
4. Convergence analysis: verify $O(h^2)$ error as the mesh refines.
5. Alternative sharp and flat combinations (DPP-flat, etc.).

---

## Mathematical Foundations

### Discrete Exterior Calculus

- Exterior forms as cochains[^hirani2003-ch3].
- Circumcentric dual complexes[^desbrun2005-s3][^hirani2003-s24].
- Hodge star via volume ratios[^hirani2003-def411].
- Sharp/flat with support volumes[^hirani2003-ch5].

### Discrete Differential Geometry

- Meyer mixed Voronoi areas for curvature[^meyer2003-s32_34].
- Cotangent Laplacian for mean curvature[^meyer2003-eq8].
- Angle defect for Gaussian curvature[^meyer2003-eq9].

### Key Theorems Preserved

- Discrete Stokes theorem (exact).
- Gauss-Bonnet theorem for closed surfaces (exact up to floating-point
  rounding).
- Conservation of dual volumes (exact up to floating-point rounding).
- The vector calculus identities $\nabla \times \nabla f = 0$ and
  $\nabla \cdot (\nabla \times v) = 0$ hold only approximately for the
  LSQ operators (see Example 3).

---

## References

The three primary works are:

1. Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
   *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*.
   In: Visualization and Mathematics III, pp. 35-57.
2. Desbrun, M., Hirani, A. N., Leok, M., & Marsden, J. E. (2005).
   *Discrete Exterior Calculus*. arXiv:math/0508341v2.
3. Hirani, A. N. (2003). *Discrete Exterior Calculus*. PhD thesis,
   California Institute of Technology.

Footnotes throughout this document point to specific sections and equations
in these works.

[^meyer2003-eq5]: Meyer et al. (2003), *Discrete Differential-Geometry
Operators for Triangulated 2-Manifolds*, Eq. 5 (cotangent weights).
[^meyer2003-eq7]: Meyer et al. (2003), Eq. 7 (circumcentric Voronoi formula
for acute triangles).
[^meyer2003-eq8]: Meyer et al. (2003), Eq. 8 (cotangent Laplacian for mean
curvature).
[^meyer2003-eq9]: Meyer et al. (2003), Eq. 9 (angle defect for Gaussian
curvature).
[^meyer2003-fig4]: Meyer et al. (2003), Fig. 4 (mixed-area subdivision for
obtuse triangles).
[^meyer2003-s32_34]: Meyer et al. (2003), §3.2-3.4 (mixed Voronoi areas).
[^desbrun2005]: Desbrun et al. (2005), *Discrete Exterior Calculus*,
arXiv:math/0508341v2.
[^desbrun2005-s3]: Desbrun et al. (2005), §3 (Primal Simplicial Complex and
Dual Cell Complex; circumcentric duals).
[^desbrun2005-s5]: Desbrun et al. (2005), §5 (Differential Forms and
Exterior Derivative).
[^desbrun2005-s6]: Desbrun et al. (2005), §6 (Hodge Star and Codifferential).
[^hirani2003-s24]: Hirani (2003), *Discrete Exterior Calculus* (PhD thesis),
§2.4 (Dual Complex; circumcentric dual cells).
[^hirani2003-def245]: Hirani (2003), Definition 2.4.5 (Circumcentric Dual
Cell).
[^hirani2003-ch3]: Hirani (2003), Chapter 3 (Discrete Forms and Exterior
Derivative).
[^hirani2003-def411]: Hirani (2003), Definition 4.1.1 (Hodge star via volume
ratios).
[^hirani2003-ch5]: Hirani (2003), Chapter 5 (Forms and Vector Fields;
sharp and flat operators).
[^hirani2003-s56]: Hirani (2003), §5.6 (PDP-flat operator).
[^hirani2003-prop553]: Hirani (2003), Proposition 5.5.3 (sharp and flat are
not exact inverses).
[^hirani2003-eq581]: Hirani (2003), Eq. 5.8.1 (PP-sharp formula).
[^hirani2003-eq642]: Hirani (2003), Eq. 6.4.2 (Laplace-Beltrami).
