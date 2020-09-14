import numpy as np
import os
import trimesh

from src.vox import load_sample


def get_scannet(scan_id, scannet_dir='datasets/scannet', output_type='vox'):
    assert output_type in ['vox', 'mesh', 'array']

    if output_type == 'vox':
        scan_path = os.path.join(scannet_dir, scan_id, scan_id + '.vox')
        vox = load_sample(scan_path)
        return vox
    else:
        scan_path = os.path.join(scannet_dir, scan_id, scan_id + '_vh_clean_2.labels.ply')
        mesh = trimesh.load_mesh(scan_path)

        if output_type == 'mesh':
            return mesh
        elif output_type == 'array':
            return np.array(mesh.vertices)
