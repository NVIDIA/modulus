# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a gallery of primitives, one column per family."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyvista as pv

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives import curves, planar, procedural, surfaces, volumes

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "primitives_gallery.png"

MESH_COLOR = "#9ecae1"
EDGE_COLOR = "#304050"
CURVE_COLOR = "#0072b2"
VIEW_VECTOR = (1.0, -1.4, 0.9)

GALLERY = {
    "curves": [
        ("curves.helix_3d", curves.helix_3d.load(height=4.0, n_points=200)),
        ("curves.spiral_2d", curves.spiral_2d.load()),
    ],
    "planar": [
        ("planar.annulus_2d", planar.annulus_2d.load(n_radial=4, n_angular=32)),
        ("planar.l_shape", planar.l_shape.load(subdivisions=4)),
    ],
    "surfaces": [
        ("surfaces.torus", surfaces.torus.load(n_major=32, n_minor=12)),
        ("surfaces.mobius_strip", surfaces.mobius_strip.load()),
    ],
    "volumes": [
        ("volumes.cube_volume", volumes.cube_volume.load(subdivisions=4)),
        ("volumes.sphere_volume", volumes.sphere_volume.load(resolution=12)),
    ],
    "procedural": [
        ("procedural.lumpy_sphere", procedural.lumpy_sphere.load(subdivisions=3)),
        ("procedural.perturbed_grid", procedural.perturbed_grid.load(n_x=9, n_y=9)),
    ],
}

plt.rcParams.update({"font.size": 13})


def _render(mesh):
    """Render one primitive: top-down in 2D, oblique in 3D, cut open if solid."""

    dataset = to_pyvista(mesh)
    plotter = pv.Plotter(window_size=(700, 700))
    plotter.enable_anti_aliasing("ssaa")
    if mesh.n_manifold_dims == 1:
        plotter.add_mesh(
            dataset, color=CURVE_COLOR, line_width=6, render_lines_as_tubes=True
        )
    else:
        if mesh.n_manifold_dims == 3:
            # Keep the cells behind a plane through the center, to show the inside.
            keep = mesh.cell_centroids[:, 1] > mesh.points[:, 1].mean()
            dataset = dataset.extract_cells(keep.nonzero().squeeze(1).numpy())
        plotter.add_mesh(
            dataset,
            color=MESH_COLOR,
            show_edges=True,
            edge_color=EDGE_COLOR,
            line_width=0.6,
            ambient=0.25,
            diffuse=0.75,
        )
    plotter.set_background("white")
    if mesh.n_spatial_dims == 2:
        plotter.view_xy()
    else:
        plotter.view_vector(VIEW_VECTOR, viewup=(0.0, 0.0, 1.0))
    plotter.reset_camera()
    plotter.camera.zoom(1.2)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def main():
    """Generate the documentation image."""

    figure, axes = plt.subplots(2, len(GALLERY), figsize=(14.0, 7.4))
    for column, (family, entries) in enumerate(GALLERY.items()):
        for row, (name, mesh) in enumerate(entries):
            axis = axes[row, column]
            axis.imshow(_render(mesh))
            axis.set_title(
                f"{name}\n{mesh.n_manifold_dims}D manifold in "
                f"{mesh.n_spatial_dims}D space",
                fontsize=12.5,
            )
            axis.set_axis_off()
        axes[0, column].annotate(
            family,
            xy=(0.5, 1.32),
            xycoords="axes fraction",
            ha="center",
            fontsize=16,
            fontweight="bold",
        )
    figure.suptitle("physicsnemo.mesh.primitives", fontsize=17, y=0.995)
    figure.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.06, hspace=0.25
    )
    figure.savefig(OUTPUT, dpi=160, facecolor="white")
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
