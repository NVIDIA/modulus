# PhysicsNeMo-Mesh Primitives

`physicsnemo.mesh.primitives` is a collection of canonical meshes for
tutorials, testing, and experimentation. Each mesh module generates its mesh
at runtime with a `load()` function. The modules are grouped by family and by
dimensional configuration: the manifold dimension of the cells, and the
dimension of the space they live in (written `2D→3D` for a surface in 3D).
For the API reference, see the
[primitives page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/primitives.html).

The examples on this page run in order: the first block defines the names that
the later blocks use.

## Quick Start

```python
import torch
from physicsnemo.mesh import primitives

# Load a simple sphere
mesh = primitives.surfaces.sphere_icosahedral.load(radius=1.0, subdivisions=2)

# Load the Stanford bunny
mesh = primitives.pyvista_datasets.bunny.load()

# Create a lumpy sphere with noise
mesh = primitives.procedural.lumpy_sphere.load(noise_amplitude=0.1)
```

The `pyvista_datasets` modules, `volumes.sphere_volume`, and
`volumes.cylinder_volume` need the optional `pyvista` package. The `text`
module needs `matplotlib`.

## Categories

### basic/ - Minimal Test Meshes

Simple meshes with one or few cells for unit testing and validation.

| Function | Dimensions | Description |
| ---------- | ------------ | ------------- |
| `single_point_2d` | 0D→2D | Single point in 2D space |
| `single_point_3d` | 0D→3D | Single point in 3D space |
| `three_points_2d` | 0D→2D | Three points in 2D space |
| `three_points_3d` | 0D→3D | Three points in 3D space |
| `single_edge_2d` | 1D→2D | Single edge in 2D space |
| `single_edge_3d` | 1D→3D | Single edge in 3D space |
| `three_edges_2d` | 1D→2D | Polyline with three edges in 2D |
| `three_edges_3d` | 1D→3D | Polyline with three edges in 3D |
| `single_triangle_2d` | 2D→2D | Single triangle in 2D space |
| `single_triangle_3d` | 2D→3D | Single triangle in 3D space |
| `two_triangles_2d` | 2D→2D | Two triangles sharing an edge (2D) |
| `two_triangles_3d` | 2D→3D | Two triangles sharing an edge (3D) |
| `single_tetrahedron` | 3D→3D | Single tetrahedron |
| `two_tetrahedra` | 3D→3D | Two tetrahedra sharing a face |

### curves/ - 1D Manifolds

Curves embedded in 1D, 2D, and 3D spaces.

| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| `line_segment_1d` | 1D→1D | Line segment on real line | Open |
| `line_segments_1d` | 1D→1D | Multiple disconnected segments | Disconnected |
| `straight_line_2d` | 1D→2D | Straight line in 2D | Open |
| `circular_arc_2d` | 1D→2D | Circular arc in 2D | Open, constant curvature |
| `circle_2d` | 1D→2D | Closed circle in 2D | Closed, constant curvature |
| `ellipse_2d` | 1D→2D | Closed ellipse in 2D | Closed, variable curvature |
| `polyline_2d` | 1D→2D | Zigzag polyline in 2D | Open, piecewise linear |
| `spiral_2d` | 1D→2D | Archimedean spiral in 2D | Open |
| `straight_line_3d` | 1D→3D | Straight line in 3D | Open |
| `helix_3d` | 1D→3D | Helical curve in 3D | Open, constant curvature |
| `circle_3d` | 1D→3D | Circle in 3D (any orientation) | Closed |
| `trefoil_knot_3d` | 1D→3D | Trefoil knot in 3D | Closed, knotted |
| `spline_3d` | 1D→3D | PyVista's `load_spline` curve, built without PyVista | Open |

### planar/ - 2D Manifolds in 2D Space

Triangulated 2D shapes in the plane.

| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| `unit_square` | 2D→2D | Unit square with subdivisions | Has boundary |
| `rectangle` | 2D→2D | Rectangular domain | Has boundary |
| `equilateral_triangle` | 2D→2D | Equilateral triangle | Has boundary |
| `regular_polygon` | 2D→2D | Regular n-sided polygon | Has boundary |
| `circle_2d` | 2D→2D | Filled disk | Has boundary |
| `annulus_2d` | 2D→2D | Ring (annulus) | Has inner/outer boundaries |
| `l_shape` | 2D→2D | L-shaped domain | Non-convex, has boundary |
| `structured_grid` | 2D→2D | Regular triangular grid | Has boundary |

### surfaces/ - 2D Manifolds in 3D Space

Surface meshes embedded in 3D.

<!-- markdownlint-disable MD013 -->
| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| **Spheres** | | | |
| `sphere_icosahedral` | 2D→3D | Sphere via icosahedron subdivision | Closed, uniform triangulation |
| `sphere_uv` | 2D→3D | Sphere via lat/long (UV) parametrization | Closed, polar singularities |
| **Cylinders** | | | |
| `cylinder` | 2D→3D | Cylinder with caps | Closed |
| `cylinder_open` | 2D→3D | Cylinder without caps | Has boundary circles |
| **Other Shapes** | | | |
| `torus` | 2D→3D | Torus (donut shape) | Closed, genus=1 |
| `plane` | 2D→3D | Flat plane | Has boundary |
| `cone` | 2D→3D | Cone with an open base | Has boundary circle |
| `disk` | 2D→3D | Flat disk | Has boundary circle |
| `hemisphere` | 2D→3D | Half sphere | Has boundary circle |
| **Platonic Solids** | | | |
| `cube_surface` | 2D→3D | Cube surface (triangulated) | Closed |
| `tetrahedron_surface` | 2D→3D | Regular tetrahedron | Closed |
| `octahedron_surface` | 2D→3D | Regular octahedron | Closed |
| `icosahedron_surface` | 2D→3D | Regular icosahedron | Closed |
| **Special Surfaces** | | | |
| `mobius_strip` | 2D→3D | Möbius strip | Non-orientable, has boundary |
<!-- markdownlint-enable MD013 -->

### volumes/ - 3D Manifolds in 3D Space

Tetrahedral volume meshes.

| Function | Dimensions | Description | Properties |
| ---------- | ------------ | ------------- | ------------ |
| `cube_volume` | 3D→3D | Tetrahedral cube mesh | Structured |
| `sphere_volume` | 3D→3D | Tetrahedral sphere mesh | Delaunay |
| `cylinder_volume` | 3D→3D | Tetrahedral cylinder mesh | Delaunay |
| `tetrahedron_volume` | 3D→3D | Single tetrahedron | Minimal volume mesh |

### procedural/ - Mesh Variations and Noise Generation

Functions for creating modified versions of meshes and standalone noise generation.

Mesh variations:

<!-- markdownlint-disable MD013 -->
| Function | Description | Use Case |
| ---------- | ------------- | ---------- |
| `lumpy_sphere` | Sphere surface with radial noise (2D→3D) | Testing robustness to irregular geometry |
| `lumpy_ball` | Tetrahedral ball with radial noise (3D→3D) | Irregular volume meshes |
| `noisy_mesh` | Add Gaussian noise to any mesh | Generic perturbation utility |
| `perturbed_grid` | Structured grid with random perturbations (2D→2D) | Testing on nearly-regular grids |
<!-- markdownlint-enable MD013 -->

Procedural noise functions, in `physicsnemo.mesh.primitives.procedural.noise`:

| Function | Description | Dimensions | GPU |
| ---------- | ------------- | ------------ | ----- |
| `perlin_noise_nd` | Dimension-agnostic Perlin noise | 1D-nD | ✓ |
| `perlin_noise_1d` | 1D Perlin noise | 1D | ✓ |
| `perlin_noise_2d` | 2D Perlin noise | 2D | ✓ |
| `perlin_noise_3d` | 3D Perlin noise | 3D | ✓ |

```python
# Generate noise at mesh centroids
from physicsnemo.mesh.primitives.procedural.noise import perlin_noise_nd

centroids = mesh.cell_centroids
noise = perlin_noise_nd(centroids, scale=1.0, seed=42)
mesh.cell_data["noise"] = noise

# Works on any dimensional mesh
points_4d = torch.randn(100, 4)
noise_4d = perlin_noise_nd(points_4d, scale=2.0, seed=123)
```

### pyvista_datasets/ - PyVista Examples

Wrappers for PyVista's example datasets. PyVista downloads some of them on
first use and caches them.

| Function | Dimensions | Description |
| ---------- | ------------ | ------------- |
| `airplane` | 2D→3D | Classic airplane surface mesh |
| `bunny` | 2D→3D | Stanford bunny (computer graphics classic) |
| `ant` | 2D→3D | Ant surface mesh |
| `cow` | 2D→3D | Cow mesh (mixed cell types, auto-triangulated) |
| `globe` | 2D→3D | Earth globe surface |
| `tetbeam` | 3D→3D | Tetrahedral beam (FEA test case) |
| `hexbeam` | 3D→3D | Hexahedral beam (auto-tessellated to tets) |

### text - Text as Meshes

`physicsnemo.mesh.primitives.text` renders a string into a mesh with
matplotlib's fonts: `text_1d_2d` (outlines), `text_2d_2d` (filled glyphs),
`text_2d_3d` (the surface of extruded glyphs), and `text_3d_3d` (extruded
volume).

```python
from physicsnemo.mesh.primitives.text import text_2d_2d

label = text_2d_2d("mesh")
```

## Dimensional Coverage

The primitives cover these dimensional configurations:

- 0D→2D, 0D→3D: point clouds (in `basic`)
- 1D→1D: line segments on the real number line
- 1D→2D: curves in the plane (circles, spirals, etc.)
- 1D→3D: space curves (helix, knots, etc.)
- 2D→2D: planar triangulations (squares, circles, polygons)
- 2D→3D: surface meshes (spheres, cylinders, tori, etc.)
- 3D→3D: volume meshes (tetrahedral solids)

## Usage Patterns

### Basic Loading

```python
# Every mesh module has a load() function
mesh = primitives.surfaces.sphere_icosahedral.load(device="cpu")
```

### Parametric Control

```python
# Adjust resolution
sphere_coarse = primitives.surfaces.sphere_icosahedral.load(subdivisions=1)
sphere_fine = primitives.surfaces.sphere_icosahedral.load(subdivisions=4)

# Adjust size
cylinder_small = primitives.surfaces.cylinder.load(radius=0.5, height=1.0)
cylinder_large = primitives.surfaces.cylinder.load(radius=2.0, height=5.0)
```

### Device Selection

```python
# CPU
mesh_cpu = primitives.pyvista_datasets.bunny.load(device="cpu")

# GPU, if one is present
if torch.cuda.is_available():
    mesh_gpu = primitives.pyvista_datasets.bunny.load(device="cuda")
```

### Procedural Variations

```python
# Create base mesh
base_sphere = primitives.surfaces.sphere_icosahedral.load(subdivisions=3)

# Add noise
noisy_sphere = primitives.procedural.noisy_mesh.load(
    base_mesh=base_sphere, noise_scale=0.05, seed=42
)

# Or use pre-made lumpy sphere
lumpy = primitives.procedural.lumpy_sphere.load(noise_amplitude=0.1, seed=42)
```

## Design Principles

1. One file per mesh: each mesh type is in its own module.
2. Consistent interface: every mesh module has a `load()` function. (The
   `text` module and `procedural.noise` provide functions instead.)
3. Runtime generation: no stored mesh files (PyVista caches its own
   downloads).
4. Parametric: most meshes accept resolution and size parameters.
5. Device-aware: every `load()` accepts `device`, except
   `procedural.noisy_mesh`, which keeps the device of its `base_mesh`.
6. Documented: each module states its manifold and spatial dimensions.

## Testing

The physicsnemo.mesh test suite uses these primitives throughout. You can also
use them in your own tests:

```python
def test_my_algorithm():
    from physicsnemo.mesh import primitives

    # Test on various mesh types
    for mesh_loader in [
        primitives.surfaces.sphere_icosahedral.load,
        primitives.pyvista_datasets.bunny.load,
        primitives.surfaces.torus.load,
    ]:
        mesh = mesh_loader()
        result = my_algorithm(mesh)
        assert result is not None
```
