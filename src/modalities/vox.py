# Vox - A grid-based signed distance field (level set) class.
# Written by Armen Avetisyan (https://github.com/skanti/Scan2CAD/blob/master/Network/base/Vox.py)

import os
import struct
import numpy as np

from src.geometry.transformations import add_forth_coord, apply_transform


class Vox:
    def __init__(self, dims=[0, 0, 0], res=0, grid2world=None, sdf=None, pdf=None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.grid2world = grid2world
        self.sdf = sdf
        self.pdf = pdf

    @classmethod
    def load_sample(cls, filename):
        assert os.path.isfile(filename), "file not found: %s" % filename
        if filename.endswith(".df"):
            f_or_c = "C"
        else:
            f_or_c = "F"

        fin = open(filename, 'rb')

        s = cls()
        s.filename = filename
        s.dimx = struct.unpack('I', fin.read(4))[0]
        s.dimy = struct.unpack('I', fin.read(4))[0]
        s.dimz = struct.unpack('I', fin.read(4))[0]
        s.res = struct.unpack('f', fin.read(4))[0]
        n_elems = s.dimx * s.dimy * s.dimz

        s.grid2world = struct.unpack('f' * 16, fin.read(16 * 4))
        sdf_bytes = fin.read(n_elems * 4)
        try:
            s.sdf = struct.unpack('f' * n_elems, sdf_bytes)
        except struct.error:
            print("Cannot load", filename)
            s.sdf = np.ones((1, s.dimz, s.dimy, s.dimx), dtype=np.float32) * -0.15

        pdf_bytes = fin.read(n_elems * 4)
        if pdf_bytes:
            s.pdf = struct.unpack('f' * n_elems, pdf_bytes)
        fin.close()
        s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order=f_or_c)
        s.sdf = np.asarray(s.sdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])
        if pdf_bytes:
            s.pdf = np.asarray(s.pdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])
        else:
            s.pdf = np.zeros((1, s.dimz, s.dimy, s.dimx), dtype=np.float32)

        return s

    def write(self, filename):
        with open(filename, 'wb') as fout:
            fout.write(struct.pack('I', self.dimx))
            fout.write(struct.pack('I', self.dimy))
            fout.write(struct.pack('I', self.dimz))
            fout.write(struct.pack('f', self.res))

            n_elems = self.dimx * self.dimy * self.dimz
            fout.write(struct.pack('f' * 16, *self.grid2world.flatten('F')))
            fout.write(struct.pack('f' * n_elems, *self.sdf.flatten('C')))
            if self.pdf is not None:
                fout.write(struct.pack('f'*n_elems, *self.pdf.flatten('C')))

    def to_pointcloud(self, res=0.1, apply_grid2world=True, return_coord_dim=3, return_index=False):
        """
        Translate VOX format to list of vertices

        :param res: border for surface |SDF| <= res
        :param apply_grid2world: if True, then apply grid2world Transformation INDEX -> Coords
        :param return_coord_dim: 3 or 4 output coordinate dimension (forth coordinate is const = 1)
        :param return_index: if True, then also return indices of surface cells
        :return: np.array of vertices
        """
        assert return_coord_dim == 3 or return_coord_dim == 4

        indices = np.argwhere(np.abs(self.sdf[0]) <= res)
        vertices = add_forth_coord(indices[:, ::-1])

        if apply_grid2world:
            vertices = apply_transform(vertices, [self.grid2world])

        vertices = vertices[:, :return_coord_dim]
        if return_index:
            return vertices, indices
        else:
            return vertices
