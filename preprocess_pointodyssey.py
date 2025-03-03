# This script splits the anno.npz files into smaller files for each frame
# The annotations can then quickly be loaded for each frame instead of loading the entire file every time/preloading everything into RAM
import os
import sys
from tqdm import trange
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python prepare_point_odyssey.py <root>")
    sys.exit(1)

root = sys.argv[1]

for split in ["train", "sample", "val", "test"]:
    path = os.path.join(root, split)
    if not os.path.exists(path):
        continue

    scenes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for scene in scenes:
        if os.path.exists(os.path.join(path, scene, "annotations")):
            num_frames = len(
                [
                    f
                    for f in os.listdir(os.path.join(path, scene, "rgbs"))
                    if f.endswith(".jpg")
                ]
            )
            num_annotations = len(
                [
                    f
                    for f in os.listdir(os.path.join(path, scene, "annotations"))
                    if f.endswith(".npz")
                ]
            )
            if num_frames == num_annotations:
                print(f"Annotations already split for {scene} in {split} split")
                continue
        os.makedirs(os.path.join(path, scene, "annotations"), exist_ok=True)
        annotations = np.load(os.path.join(path, scene, "anno.npz"))

        trajs_2d = annotations["trajs_2d"]
        trajs_3d = annotations["trajs_3d"]
        valids = annotations["valids"]
        visibs = annotations["visibs"]
        intrinsics = annotations["intrinsics"]
        extrinsics = annotations["extrinsics"]

        N = trajs_2d.shape[0]
        print(f"Splitting {N} frames for {scene} in {split} split")
        print(f"Shape of trajs_2d: {trajs_2d.shape}")
        print(f"Shape of trajs_3d: {trajs_3d.shape}")
        print(f"Shape of valids: {valids.shape}")
        print(f"Shape of visibs: {visibs.shape}")
        print(f"Shape of intrinsics: {intrinsics.shape}")
        print(f"Shape of extrinsics: {extrinsics.shape}")
        for i in trange(N):
            np.savez_compressed(
                os.path.join(path, scene, "annotations", f"anno_{i:05d}.npz"),
                trajs_2d=trajs_2d[i] if len(trajs_2d.shape) > 1 else trajs_2d,
                trajs_3d=trajs_3d[i] if len(trajs_3d.shape) > 1 else trajs_3d,
                valids=valids[i] if len(valids.shape) > 1 else valids,
                visibs=visibs[i] if len(visibs.shape) > 1 else visibs,
                intrinsics=intrinsics[i] if len(intrinsics.shape) > 1 else intrinsics,
                extrinsics=extrinsics[i] if len(extrinsics.shape) > 1 else extrinsics,
            )
