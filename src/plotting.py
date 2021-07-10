import meshplot as mp
import numpy as np


def plot_ply(points, p=None, c='green', point_size=0.1, cm='viridis'):
    if not isinstance(points, np.ndarray):
        point_cloud = np.array(points.vertices)
    else:
        point_cloud = points

    if point_cloud.shape[-1] == 4:
        point_cloud = point_cloud[:, :3]

    d = {"point_size": point_size, 'colormap': cm}

    if p is None:
        p = mp.plot(point_cloud, c=c, shading=d, return_plot=True)
    else:
        p.add_points(point_cloud, c=c, shading=d)
    return p


def plot_mesh(mesh, p=None, c=None, point_size=0.1, cm='hot'):
    if c is None:
        c = np.array(mesh.vertices[:, 0])

    d = {"point_size": point_size, 'colormap': cm, 'alpha': 0.1}

    if p is None:
        p = mp.plot(mesh.vertices, mesh.faces, c=c, shading=d, return_plot=True)
    else:
        p.add_mesh(mesh.vertices, mesh.faces, c=c, shading=d)
    return p
