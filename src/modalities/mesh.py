# apply transform to mesh vertices
from src.geometry.transformations import apply_transform
from src.plotting import plot_mesh, plot_ply

import trimesh
import numpy as np


class Mesh:
    def __init__(self, mesh: trimesh.Trimesh):
        self._mesh = mesh

    @classmethod
    def load_ply(cls, filename):
        mesh = trimesh.load(filename)
        return cls(mesh)

    def transform(self, transform_list: list, apply_inverse: bool = False):
        # get mesh vertices and apply transformatio
        vertices = np.array(self._mesh.vertices)
        vertices = apply_transform(vertices, transform_list, apply_inverse)

        # set new vertices
        self._mesh.vertices = vertices

    def plot(self, previous_plot, color, point_size, colormap, vertices_only: bool = False):
        if vertices_only:
            new_plot = plot_ply(self._mesh, p=previous_plot, c=color, point_size=point_size, cm=colormap)
        else:
            new_plot = plot_mesh(self._mesh, p=previous_plot, c=color, point_size=point_size, cm=colormap)

        return new_plot
