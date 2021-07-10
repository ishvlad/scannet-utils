import numpy as np
import os
import trimesh

from src.modalities.vox import Vox


class Scannet:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir

    def load_vox(self, scan_id):
        scan_path = os.path.join(self.scannet_dir, scan_id, scan_id + '.vox')
        vox = Vox.load_sample(scan_path)
        return vox

    def load_mesh(self, scan_id, vertices_only=False):
        scan_path = os.path.join(self.scannet_dir, scan_id, scan_id + '_vh_clean_2.labels.ply')
        mesh = trimesh.load_mesh(scan_path)

        if vertices_only:
            return np.array(mesh.vertices)
        return mesh
