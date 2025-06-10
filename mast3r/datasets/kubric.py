# Modified from LocoTrack's pytorch kubric loader
# https://github.com/cvlab-kaist/locotrack/blob/main/locotrack_pytorch/data/kubric_data.py

import os
import cv2
import glob
import numpy as np
import os.path as osp

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import imread_cv2
from .utils import farthest_point_sample_py
from mast3r.datasets.base.mast3r_base_stereo_view_dataset import (
    MASt3RBaseStereoViewDataset,
)


class Kubric(MASt3RBaseStereoViewDataset):
    def __init__(
        self,
        root,
        split,
        n_corres=32,
        maximum_stride=32,
        quick=False,
        verbose=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dataset_label = "panning_movi_e"
        self.split = split
        self.S = 2  # 2 views
        self.N = n_corres  # min num points
        self.verbose = verbose
        self.maximum_stride = maximum_stride

        self.base_paths = []
        self.sample_indexes = []

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(osp.join(root, split))

        for subdir in self.subdirs:
            for seq in glob.glob(osp.join(subdir, "*/")):
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print(f"found {len(self.sequences)} unique videos in {root} (split={split})")

        ## load trajectories
        print("loading trajectories...")

        if quick:
            self.sequences = self.sequences[1:2]

        for base_path in self.sequences:
            if self.verbose:
                print(f"seq {base_path}")

            video_length = len(os.listdir(osp.join(base_path, "rgbs")))

            for i in range(1, video_length):
                self.base_paths.append(base_path)
                self.sample_indexes.append(i)

        print(f"Found {len(self.base_paths)} candidates in {root} (dset={split})")


    def __len__(self):
        return len(self.sample_indexes)

    def get_corres(self, trajs, idxs):
        points = trajs["target_points"]
        occluded = trajs["occluded"]

        pts1 = points[:, idxs[0]]
        pts2 = points[:, idxs[1]]
        occluded1 = occluded[:, idxs[0]]
        occluded2 = occluded[:, idxs[1]]

        vis_in_both = np.logical_and(~occluded1, ~occluded2)

        pts1 = pts1[vis_in_both]
        pts2 = pts2[vis_in_both]

        if len(pts1) < self.N:
            valid = np.zeros(self.N, dtype=bool)
            valid[: len(pts1)] = True
            pts1 = np.concatenate([pts1, np.zeros((self.N - len(pts1), 2))], axis=0, dtype=np.float32)
            pts2 = np.concatenate([pts2, np.zeros((self.N - len(pts2), 2))], axis=0, dtype=np.float32)

            return [pts1, pts2], valid

        inds = farthest_point_sample_py(pts1, self.N)
        pts1 = pts1[inds]
        pts2 = pts2[inds]
        valid = np.ones(self.N, dtype=bool)

        return [pts1, pts2], valid

    def _get_views(self, idx, resolution, rng):
        base_path = self.base_paths[idx]
        sample_index = self.sample_indexes[idx]
        
        tracks_path = osp.join(base_path, "tracks.npz")
        camera_path = osp.join(base_path, "camera.npz")

        lower = max(0, sample_index - self.maximum_stride)
        pair_index = np.random.randint(lower, sample_index)
        full_idx = [pair_index, sample_index]

        rgb_paths = [
            osp.join(base_path, "rgbs", f"{full_idx[0]:02d}.jpg"),
            osp.join(base_path, "rgbs", f"{full_idx[1]:02d}.jpg"),
        ]
        depth_paths = [
            osp.join(base_path, "depths", f"{full_idx[0]:02d}.png"),
            osp.join(base_path, "depths", f"{full_idx[1]:02d}.png"),
        ]

        trajs = np.load(tracks_path)
        all_corres, valid_corres = self.get_corres(trajs, full_idx)

        views = []
        camera = np.load(camera_path)
        min_depth, max_depth = camera["depth_range"]
        for i in range(2):
            rgb_image = imread_cv2(rgb_paths[i])
            depthmap = imread_cv2(depth_paths[i], options=cv2.IMREAD_ANYDEPTH)
            depthmap = depthmap / 65535 * (max_depth - min_depth) + min_depth
            depthmap = depthmap.astype(np.float32)

            idx = full_idx[i]
            intrinsics = camera["intrinsics"]
            camera_pose = camera["matrix_world"][idx]
            rgb_image, depthmap, corres, valid_corres, intrinsics = (
                self._crop_resize_if_necessary(
                    rgb_image,
                    depthmap,
                    all_corres[i],
                    valid_corres,
                    intrinsics,
                    resolution,
                    rng=rng,
                )
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    corres=corres,
                    camera_intrinsics=intrinsics,
                    dataset=self.dataset_label,
                    label=rgb_paths[i].split("/")[-3],
                    instance=osp.split(rgb_paths[i])[1],
                )
            )

        views[0]["valid_corres"] = valid_corres
        views[1]["valid_corres"] = valid_corres

        return views
