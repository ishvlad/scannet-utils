import numpy as np

from src.transformations import apply_transform, add_forth_coord


def vox2obj(vox, res=0.1, apply_grid2world=True, return_coord_dim=3, return_index=False):
    """
    Translate VOX format to list of vertices

    :param vox: VOX format (DF or SDF) of the object
    :param res: border for surface |SDF| <= res
    :param apply_grid2world: if True, then apply grid2world Transformation INDEX -> Coords
    :param return_coord_dim: 3 or 4 output coordinate dimension (forth coordinate is const = 1)
    :param return_index: if True, then also return indices of surface cells
    :return: np.array of vertices
    """
    assert return_coord_dim == 3 or return_coord_dim == 4

    indices = np.argwhere(np.abs(vox.sdf[0]) <= res)
    vertices = indices[:, ::-1]

    if apply_grid2world:
        vertices = apply_transform(vertices, vox.grid2world)

    if return_coord_dim:
        vertices = add_forth_coord(vertices)

    if return_index:
        return vertices, indices
    else:
        return vertices

