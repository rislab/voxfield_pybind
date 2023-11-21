#!/usr/bin/env python3
# Original implementation by Federico Magistri
import glob
import os

import numpy as np
import numpy.matlib
import open3d as o3d

from cache import get_cache, memoize


class CowDataset:
    def __init__(self, data_source, get_color: bool = False, apply_pose: bool = False):
        # Cache
        self.use_cache = False#True
        self.cache = get_cache(directory="cache/cow/")
        self.get_color = get_color
        self.apply_pose = apply_pose

        self.data_source = os.path.join(data_source, "")
        self.gt_list = self.read_gt_list(os.path.join(self.data_source, "poses.txt"))
        self.cloud_files = sorted(glob.glob(self.data_source + "*.ply"))

    @staticmethod
    def read_gt_list(filename):
        poses = np.loadtxt(filename, delimiter=",", dtype=np.float32)
        # poses = np.loadtxt(filename, delimiter=" ", dtype=np.float32)
        return poses.reshape((len(poses), 4, 4))

    @memoize()
    def __getitem__(self, idx):
        pose = self.gt_list[idx]
        pcd = o3d.io.read_point_cloud(self.cloud_files[idx])


        ## # inverse of pose -this is very bad
        ## R1 = pose[0:3,0:3]
        ## t1 = pose[0:3,-1]
        ## t2 = -1* (np.linalg.inv(R1)@t1)
        ## R2 = np.linalg.inv(R1)
        ## pose = np.zeros((4,4))
        ## pose[0:3,0:3] = R2
        ## pose[0:3,-1] = t2

        # my dirty hack (seems equivalent to pcd.transform)
        # pts = np.asarray(pcd.points)
        # R = pose[0:3,0:3]
        # t = pose[0:3,-1].reshape((3,1))
        # if (self.apply_pose):
        #     # print("apply pose")
        #     pts = np.matmul(R, np.transpose(pts)) + \
        #         np.matlib.repmat(t, 1, np.shape(pts)[0])
        #     pts = pts.transpose()
        # xyz = pts

        pcd.transform(pose) if self.apply_pose else None
        xyz = np.array(pcd.points)
        colors = np.array(pcd.colors)

        if self.get_color:
            return xyz, colors, pose
        return xyz, pose

    def __len__(self):
        return len(self.cloud_files)
