#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from mast3r.model import AsymmetricMASt3R
from mast3r.losses import (
    ConfMatchingLoss,
    MatchingLoss,
    APLoss,
    Regr3D,
    InfoNCE,
    Regr3D_ScaleShiftInv,
)
from mast3r.datasets import PointOdyssey, Kubric

import mast3r.utils.path_to_dust3r  # noqa

# add mast3r classes to dust3r imports
import dust3r.training
import dust3r.datasets

dust3r.training.AsymmetricMASt3R = AsymmetricMASt3R
dust3r.training.Regr3D = Regr3D
dust3r.training.Regr3D_ScaleShiftInv = Regr3D_ScaleShiftInv
dust3r.training.MatchingLoss = MatchingLoss
dust3r.training.ConfMatchingLoss = ConfMatchingLoss
dust3r.training.InfoNCE = InfoNCE
dust3r.training.APLoss = APLoss

dust3r.datasets.Kubric = Kubric
dust3r.datasets.PointOdyssey = PointOdyssey

from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from dust3r.training import train  # noqa


def get_args_parser():
    parser = dust3r_get_args_parser()
    # change defaults
    parser.prog = "MASt3R training"
    parser.set_defaults(model="AsymmetricMASt3R(patch_embed_cls='ManyAR_PatchEmbed')")
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    train(args)
