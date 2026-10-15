# Mesh Visualization

`Mesh.draw()` plots a mesh with matplotlib or PyVista, optionally colored by
point or cell data. For the API reference, see the
[visualization page](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/visualization.html).

The examples on this page run in order: the first block defines the mesh that
the later blocks use.

## Quick Start

```python
import torch
from physicsnemo.mesh import Mesh

# Create a simple mesh: two triangles in 2D
points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
mesh = Mesh(points=points, cells=cells)

# Draw with automatic backend selection
mesh.draw()
```

## Backend Selection

With `backend="auto"` (the default), the backend depends on the spatial
dimension of the mesh (`n_spatial_dims`):

- 1D and 2D space: matplotlib.
- 3D space: PyVista.

If the preferred backend is not installed, `draw()` uses the other one.
Meshes in more than three spatial dimensions raise a `ValueError`.

You can also select the backend explicitly:

```python
# Force matplotlib (supports 3D via mplot3d)
mesh.draw(backend="matplotlib")

# Force PyVista
mesh.draw(backend="pyvista")
```

## Scalar Data Visualization

### Basic Usage

Color points or cells using scalar data:

```python
# Color cells by pressure
mesh.cell_data["pressure"] = torch.rand(mesh.n_cells)
mesh.draw(cell_scalars="pressure", cmap="coolwarm")

# Color points by temperature
mesh.point_data["temperature"] = torch.rand(mesh.n_points)
mesh.draw(point_scalars="temperature", cmap="plasma")
```

### Direct Tensor Values

Pass scalar values directly without storing them:

```python
# Generate scalar values on the fly
scalars = torch.randn(mesh.n_cells)
mesh.draw(cell_scalars=scalars)
```

### Vector Fields (Automatic Norming)

`draw()` colors by the L2 norm of multi-dimensional data:

```python
# 3D velocity field at each point
mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3)

# Displays the norm: sqrt(vx² + vy² + vz²)
mesh.draw(point_scalars="velocity")
```

### Nested TensorDict Keys

Access nested data with tuple keys:

```python
from tensordict import TensorDict

mesh.cell_data["flow"] = TensorDict(
    {
        "temperature": torch.rand(mesh.n_cells),
        "pressure": torch.rand(mesh.n_cells),
    },
    batch_size=[mesh.n_cells],
)

# Access nested data
mesh.draw(cell_scalars=("flow", "temperature"))
```

## Visualization Parameters

### Colormap Control

```python
mesh.draw(
    cell_scalars="pressure",
    cmap="viridis",  # Colormap name
    vmin=0.0,  # Minimum value
    vmax=1.0,  # Maximum value
)
```

### Transparency Control

```python
mesh.draw(
    alpha_points=1.0,  # Point opacity (0-1)
    alpha_cells=0.3,  # Cell opacity (0-1)
    alpha_edges=0.7,  # Edge opacity (0-1); matplotlib only
    show_edges=True,  # Display cell edges
)
```

The PyVista backend ignores `alpha_edges` and warns when it is not 1.0.

### Custom Axes (Matplotlib Only)

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
mesh.draw(backend="matplotlib", ax=ax, show=False)
ax.set_title("Custom Title")
plt.show()
```

## Color Logic

The neutral colors depend on which scalars are active:

- Points: black when `point_scalars=None`.
- Cells:
  - No scalars (`point_scalars=None`, `cell_scalars=None`): light blue.
  - Point scalars active: colored by the point scalars. Matplotlib fills each
    triangle with the mean of its vertex colors. PyVista interpolates the point
    scalars across each cell. Line segments that matplotlib draws in 1D or 3D
    space stay light gray.
  - Cell scalars active: colored by the cell scalars.

## Mutual Exclusivity

`point_scalars` and `cell_scalars` are mutually exclusive, so that one
colormap describes the plot:

```python
# Valid
mesh.draw(point_scalars="temperature")
mesh.draw(cell_scalars="pressure")
mesh.draw()  # No scalars

# Invalid: raises ValueError
try:
    mesh.draw(point_scalars="temperature", cell_scalars="pressure")
except ValueError as error:
    print(error)  # point_scalars and cell_scalars are mutually exclusive.
```

## Return Values

`draw()` returns the axes or plotter whether or not it shows the plot.

```python
# Matplotlib: returns a matplotlib Axes
ax = mesh.draw(backend="matplotlib", show=False)
ax.set_title("My Mesh")

# PyVista: returns a pyvista Plotter
plotter = mesh.draw(backend="pyvista", show=False)
plotter.camera_position = "xy"
plotter.show()
```

## Complete API Reference

The signature of `Mesh.draw()`, with its defaults:

```python
mesh.draw(
    backend="auto",  # "auto", "matplotlib", "pyvista"
    show=True,  # Display immediately
    point_scalars=None,  # Point scalar data: tensor, key, or nested key
    cell_scalars=None,  # Cell scalar data: tensor, key, or nested key
    cmap="viridis",  # Colormap name
    vmin=None,  # Colormap min value
    vmax=None,  # Colormap max value
    alpha_points=1.0,  # Point opacity
    alpha_cells=1.0,  # Cell opacity
    alpha_edges=1.0,  # Edge opacity (matplotlib only)
    show_edges=True,  # Display edges
    ax=None,  # Existing matplotlib Axes or pyvista Plotter to draw on
    backend_options=None,  # Dict of extra arguments for PyVista's add_mesh()
)
```

The matplotlib backend ignores `backend_options` and warns when it is set.

## Examples

### 2D Triangle Mesh

```python
# Create mesh
points = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
mesh = Mesh(points=points, cells=cells)

# Add data and visualize
mesh.cell_data["pressure"] = torch.tensor([1.0, 1.5])
mesh.point_data["temperature"] = points[:, 0] + points[:, 1]
mesh.draw(cell_scalars="pressure", cmap="coolwarm", show_edges=True)
```

### Multiple Visualizations

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: point scalars
mesh.draw(
    backend="matplotlib",
    ax=axes[0],
    show=False,
    point_scalars="temperature",
)
axes[0].set_title("Point Scalars")

# Right: cell scalars
mesh.draw(
    backend="matplotlib",
    ax=axes[1],
    show=False,
    cell_scalars="pressure",
)
axes[1].set_title("Cell Scalars")

plt.tight_layout()
plt.show()
```

### 3D Surface Mesh

```python
from physicsnemo.mesh.io import from_pyvista
import pyvista as pv

# Load an example surface
pv_mesh = pv.examples.load_airplane()
surface = from_pyvista(pv_mesh)

# Visualize with PyVista, with extra arguments for add_mesh()
surface.draw(
    alpha_cells=0.8,
    show_edges=True,
    backend_options={"smooth_shading": True},
)
```

## Implementation Details

The visualization code has four files:

- `physicsnemo/mesh/visualization/draw_mesh.py`: main dispatcher with backend
  selection.
- `physicsnemo/mesh/visualization/_matplotlib_impl.py`: matplotlib backend
  (1D to 3D space; no tetrahedra).
- `physicsnemo/mesh/visualization/_pyvista_impl.py`: PyVista backend.
- `physicsnemo/mesh/visualization/_scalar_utils.py`: scalar data processing.

Features:

- L2 norm of multi-dimensional data.
- Nested TensorDict key lookup.
- Neutral colors chosen from the active scalars.
- Validation of the scalar arguments.
- 3D plots in matplotlib via `mpl_toolkits.mplot3d`.
