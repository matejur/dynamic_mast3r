# Modified from PIP's PointOdyssey dataset loader
# https://github.com/aharley/pips2/blob/master/datasets/pointodysseydataset.py
# And from Monst3r's PointOdyssey dataset loader
# https://github.com/Junyi42/monst3r/blob/main/dust3r/datasets/pointodyssey.py

import os
import sys
import glob
import os.path as osp

import cv2
import numpy as np

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import imread_cv2
from .utils import farthest_point_sample_py
from mast3r.datasets.base.mast3r_base_stereo_view_dataset import (
    MASt3RBaseStereoViewDataset,
)


class PointOdyssey(MASt3RBaseStereoViewDataset):
    def __init__(
        self,
        root,
        split,
        n_corres=32,
        maximum_stride=10,
        clip_step=2,
        quick=False,
        verbose=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dataset_label = "pointodyssey"
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
        print(
            "found %d unique videos in %s (split=%s)"
            % (len(self.sequences), root, split)
        )

        ## load trajectories
        print("loading trajectories...")

        if quick:
            self.sequences = self.sequences[1:2]

        for base_path in self.sequences:
            if self.verbose:
                print(f"seq {base_path}")

            annotations_path = os.path.join(seq, "anno.npz")
            if os.path.isfile(annotations_path):
                video_length = len(os.listdir(osp.join(base_path, "rgbs")))

                for i in range(1, video_length, clip_step):
                    self.base_paths.append(base_path)
                    self.sample_indexes.append(i)
            else:
                print("missing annotations:", annotations_path)

        print(f"Found {len(self.base_paths)} candidates in {root} (dset={split})")

    def __len__(self):
        return len(self.sample_indexes)

    def get_corres(self, annot1, annot2, mask):
        trajs = np.stack([annot1["trajs_2d"], annot2["trajs_2d"]], axis=0)
        visibs = np.stack([annot1["visibs"], annot2["visibs"]], axis=0)
        valids = np.stack([annot1["valids"], annot2["valids"]], axis=0)

        # some data is valid in 3d but invalid in 2d
        # here we will filter to the data which is valid in 2d
        valids_xy = np.ones_like(trajs)
        inf_idx = np.where(np.isinf(trajs))
        trajs[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs))
        trajs[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2) < 2)  # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs = trajs[:, vis0]
        visibs = visibs[:, vis0]
        valids = valids[:, vis0]

        S, N, D = trajs.shape
        assert D == 2
        assert S == self.S

        H, W = mask.shape

        # discard pixels that are OOB on frame0
        for si in range(S):
            oob_inds = np.logical_or(
                np.logical_or(trajs[si, :, 0] < 0, trajs[si, :, 0] > W - 1),
                np.logical_or(trajs[si, :, 1] < 0, trajs[si, :, 1] > H - 1),
            )
            visibs[si, oob_inds] = 0

        vis0 = visibs[0] > 0
        trajs = trajs[:, vis0]
        visibs = visibs[:, vis0]
        valids = valids[:, vis0]

        # discard trajs that begin exactly on segmentation boundaries
        # since their labels are ambiguous
        if np.sum(mask == 0) > 128:
            # fill holes caused by fog/smoke
            mask_filled = cv2.medianBlur(mask, 7)
            mask[mask == 0] = mask_filled[mask == 0]

        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.Canny(mask, 1, 1)
        # block apparent edges from fog/smoke
        keep = 1 - cv2.dilate((mask == 0).astype(np.uint8), kernel, iterations=1)
        edge = edge * keep
        edge = cv2.dilate(edge, kernel)

        x0, y0 = trajs[0, :, 0].astype(np.int32), trajs[0, :, 1].astype(np.int32)
        on_edge = edge[y0, x0] > 0
        trajs = trajs[:, ~on_edge]
        visibs = visibs[:, ~on_edge]
        valids = valids[:, ~on_edge]

        # update visibility annotations
        for si in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs[si, :, 0] < 1, trajs[si, :, 0] > W - 2),
                np.logical_or(trajs[si, :, 1] < 1, trajs[si, :, 1] > H - 2),
            )
            visibs[si, oob_inds] = 0

            # when a point moves far oob, don't supervise with it
            very_oob_inds = np.logical_or(
                np.logical_or(trajs[si, :, 0] < -64, trajs[si, :, 0] > W + 64),
                np.logical_or(trajs[si, :, 1] < -64, trajs[si, :, 1] > H + 64),
            )
            valids[si, very_oob_inds] = 0

        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs = trajs[:, vis0]
        visibs = visibs[:, vis0]
        valids = valids[:, vis0]

        # ensure that the point is good in frame1
        vis_and_val = valids * visibs
        vis1 = vis_and_val[1] > 0
        trajs = trajs[:, vis1]
        visibs = visibs[:, vis1]
        valids = valids[:, vis1]

        N = trajs.shape[1]
        if N < self.N:
            if self.verbose:
                print("N=%d; ideally we want N=%d, but we will pad" % (N, self.N))

        if N == 0:
            corres = np.zeros([self.S, self.N, 2])
            valid_corres = np.zeros(self.N, dtype=bool)

            return corres, valid_corres

        # even out the distribution, across initial positions and velocities
        # fps based on xy0 and mean motion
        xym = np.concatenate(
            [trajs[0], np.mean(trajs[1:] - trajs[:-1], axis=0)], axis=-1
        )
        inds = farthest_point_sample_py(xym, self.N)
        trajs = trajs[:, inds]
        visibs = visibs[:, inds]
        valids = valids[:, inds]

        N = trajs.shape[1]
        N_ = min(N, self.N)
        inds = np.random.choice(N, N_, replace=False)

        # prep for batching, by fixing N
        trajs_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
        visibs_full = np.zeros((self.S, self.N)).astype(np.float32)
        valids_full = np.zeros((self.S, self.N)).astype(np.float32)
        trajs_full[:, :N_] = trajs[:, inds]
        visibs_full[:, :N_] = visibs[:, inds]
        valids_full[:, :N_] = valids[:, inds]

        valid_corres = valids_full[0] * valids_full[1] * visibs_full[0] * visibs_full[1]

        return trajs_full, valid_corres.astype(bool)

    def _get_views(self, index, resolution, rng):
        base_path = self.base_paths[index]
        sample_index = self.sample_indexes[index]

        lower = max(0, sample_index - self.maximum_stride)
        pair_index = np.random.randint(lower, sample_index)
        full_idx = [pair_index, sample_index]

        rgb_paths = [
            osp.join(base_path, "rgbs", f"rgb_{full_idx[0]:05d}.jpg"),
            osp.join(base_path, "rgbs", f"rgb_{full_idx[1]:05d}.jpg"),
        ]
        depth_paths = [
            osp.join(base_path, "depths", f"depth_{full_idx[0]:05d}.png"),
            osp.join(base_path, "depths", f"depth_{full_idx[1]:05d}.png"),
        ]
        mask_paths = [
            osp.join(base_path, "masks", f"mask_{full_idx[0]:05d}.png"),
            osp.join(base_path, "masks", f"mask_{full_idx[1]:05d}.png"),
        ]
        annotations_path = [
            osp.join(base_path, "annotations", f"anno_{full_idx[0]:05d}.npz"),
            osp.join(base_path, "annotations", f"anno_{full_idx[1]:05d}.npz"),
        ]

        if not os.path.exists(annotations_path[0]):
            raise ValueError(
                f"Annotation file not found: {annotations_path[0]}. Did you run the preprocessing script?"
            )

        annot1 = np.load(annotations_path[0], allow_pickle=True)
        annot2 = np.load(annotations_path[1], allow_pickle=True)

        annots = [annot1, annot2]

        mask = imread_cv2(mask_paths[0], options=cv2.IMREAD_GRAYSCALE)
        all_corres, valid_corres = self.get_corres(annot1, annot2, mask)

        views = []
        for i in range(2):
            depthpath = depth_paths[i]
            annotations = annots[i]
            pix_T_cams = annotations["intrinsics"].astype(np.float32)
            world_T_cams = annotations["extrinsics"].astype(np.float32)

            # load camera params
            extrinsics = world_T_cams
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]
            camera_pose = np.eye(4, dtype=np.float32)  # cam_2_world
            camera_pose[:3, :3] = R.T
            camera_pose[:3, 3] = -R.T @ t
            intrinsics = pix_T_cams

            # load image and depth
            rgb_image = imread_cv2(rgb_paths[i])

            depth16 = cv2.imread(depthpath, cv2.IMREAD_ANYDEPTH)
            depthmap = (
                depth16.astype(np.float32) / 65535.0 * 1000.0
            )  # 1000 is the max depth in the dataset

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
