# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import numpy as np


def farthest_point_sample_py(xyz, npoint):
    N, C = xyz.shape
    inds = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype=np.int32)
    for i in range(npoint):
        inds[i] = farthest
        centroid = xyz[farthest, :].reshape(1, C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        if npoint > N:
            # if we need more samples, make them random
            distance += np.random.randn(*distance.shape)
    return inds


def get_stride_distribution(strides, dist_type="uniform"):
    # input strides sorted by descreasing order by default

    if dist_type == "uniform":
        dist = np.ones(len(strides)) / len(strides)
    elif dist_type == "exponential":
        lambda_param = 1.0
        dist = np.exp(-lambda_param * np.arange(len(strides)))
    elif dist_type.startswith("linear"):  # e.g., linear_1_2
        try:
            start, end = map(float, dist_type.split("_")[1:])
            dist = np.linspace(start, end, len(strides))
        except ValueError:
            raise ValueError(f"Invalid linear distribution format: {dist_type}")
    else:
        raise ValueError("Unknown distribution type %s" % dist_type)

    # normalize to sum to 1
    return dist / np.sum(dist)
