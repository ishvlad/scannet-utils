# Author: Angela Dai
# Link: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py

import struct
import numpy as np
import tqdm
import imageio
import zlib
import cv2


def unpack_float_matrix(file_handle):
    result = np.asarray(struct.unpack('f' * 16, file_handle.read(16 * 4)), dtype=np.float32)
    return result.reshape(4, 4)


class RGBDFrame:
    def __init__(self, file_handle):
        self.camera_to_world = unpack_float_matrix(file_handle)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(struct.unpack('c' * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(struct.unpack('c' * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    @property
    def image(self):
        return np.array(imageio.imread(self.color_data)) / 255.

    def get_depth(self, depth_resolution, color_resolution, depth_shift):
        d_h, d_w = depth_resolution
        c_h, c_w = color_resolution

        depth = zlib.decompress(self.depth_data)
        depth = np.frombuffer(depth, dtype=np.uint16).reshape(d_h, d_w)
        depth = cv2.resize(depth, (c_w, c_h), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(float) / depth_shift
        depth = depth.reshape(depth.shape[0], depth.shape[1], 1)

        return depth


class SensorData:
    COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
    COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}

    def __init__(self, filename, verbose=True):
        self.verbose = verbose
        self.version = 4

        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(struct.unpack('c' * strlen, f.read(strlen))).decode("utf-8")
            self.intrinsic_color = unpack_float_matrix(f)
            self.extrinsic_color = unpack_float_matrix(f)
            self.intrinsic_depth = unpack_float_matrix(f)
            self.extrinsic_depth = unpack_float_matrix(f)
            self.color_compression_type = self.COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = self.COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []

            if self.verbose:
                array = tqdm.tqdm(range(num_frames))
            else:
                array = range(num_frames)

            for _ in array:
                frame = RGBDFrame(f)
                self.frames.append(frame)
