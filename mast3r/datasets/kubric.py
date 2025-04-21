# Modified from LocoTrack's pytorch kubric loader
# https://github.com/cvlab-kaist/locotrack/blob/main/locotrack_pytorch/data/kubric_data.py

import os
import cv2
import sys
import glob
import numpy as np
import os.path as osp

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import imread_cv2
from .utils import get_stride_distribution, farthest_point_sample_py
from mast3r.datasets.base.mast3r_base_stereo_view_dataset import (
    MASt3RBaseStereoViewDataset,
)


class Kubric(MASt3RBaseStereoViewDataset):
    def __init__(
        self,
        root,
        split,
        n_corres=32,
        strides=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        dist_type="linear_1_2",
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

        self.rgb_paths = []
        self.depth_paths = []
        self.traj_paths = []
        self.tracks_path = []
        self.camera_path = []
        self.full_idxs = []
        self.sample_stride = []
        self.mask_paths = []
        self.strides = strides

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

        for seq in self.sequences:
            if self.verbose:
                print(f"seq {seq}")

            rgb_path = osp.join(seq, "rgbs")

            for stride in strides:
                for ii in range(0, len(os.listdir(rgb_path)) - stride, 1):
                    full_idx = ii + np.arange(self.S) * stride
                    self.rgb_paths.append(
                        [osp.join(seq, "rgbs", f"{idx:02d}.jpg") for idx in full_idx]
                    )
                    self.depth_paths.append(
                        [osp.join(seq, "depths", f"{idx:02d}.png") for idx in full_idx]
                    )
                    self.mask_paths.append(
                        [osp.join(seq, "masks", f"{idx:02d}.png") for idx in full_idx]
                    )
                    self.tracks_path.append(osp.join(seq, "tracks.npz"))
                    self.camera_path.append(osp.join(seq, "camera.npz"))
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)
                if self.verbose:
                    sys.stdout.write(".")
                    sys.stdout.flush()

        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print("stride counts:", self.stride_counts)

        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print(
            f"collected {len(self.rgb_paths)} clips of length {self.S} in {root} (dset={split})"
        )

    def __len__(self):
        return len(self.rgb_paths)

    def _resample_clips(self, strides, dist_type):
        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [
            min(self.stride_counts[stride], int(dist[i] * max_num_clips))
            for i, stride in enumerate(strides)
        ]
        print("resampled_num_clips_each_stride:", num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(
                self.stride_idxs[stride], num_clips_each_stride[i], replace=False
            ).tolist()

        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.camera_path = [self.camera_path[i] for i in resampled_idxs]
        self.tracks_path = [self.tracks_path[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

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
            pts1 = np.concatenate([pts1, np.zeros((self.N - len(pts1), 2))], axis=0)
            pts2 = np.concatenate([pts2, np.zeros((self.N - len(pts2), 2))], axis=0)

            return [pts1, pts2], valid

        inds = farthest_point_sample_py(pts1, self.N)
        pts1 = pts1[inds]
        pts2 = pts2[inds]
        valid = np.ones(self.N, dtype=bool)

        return [pts1, pts2], valid

    def _get_views(self, idx, resolution, rng):
        rgb_paths = self.rgb_paths[idx]
        depth_paths = self.depth_paths[idx]
        camera_path = self.camera_path[idx]
        tracks_path = self.tracks_path[idx]
        masks_path = self.mask_paths[idx]  # READ WITH PIL?
        full_idx = self.full_idxs[idx]

        trajs = np.load(tracks_path)
        all_corres, valid_corres = self.get_corres(trajs, full_idx)

        views = []
        camera = np.load(camera_path)
        min_depth, max_depth = camera["depth_range"]
        for i in range(2):
            rgb_image = imread_cv2(rgb_paths[i])
            depthmap = imread_cv2(depth_paths[i], options=cv2.IMREAD_ANYDEPTH)
            depthmap = depthmap / 65535 * (max_depth - min_depth) + min_depth

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
