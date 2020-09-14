import meshplot as mp
import numpy as np


def plot_ply(mesh, p=None, c='green', point_size=0.1):
    if not isinstance(mesh, np.ndarray):
        point_cloud = np.array(mesh.vertices)
    else:
        point_cloud = mesh

    if point_cloud.shape[-1] == 4:
        point_cloud = point_cloud[:, :3]

    if p is None:
        p = mp.plot(point_cloud, c=c, shading={"point_size": point_size}, return_plot=True)
    else:
        p.add_points(point_cloud, c=c, shading={"point_size": point_size})
    return p
