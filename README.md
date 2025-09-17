# Dynamic MASt3R

This repository contains small changes to the original [MASt3R](https://github.com/naver/mast3r) for training on dynamic scenes.
It is then used for point tracking as part of my master's thesis and not really on its own.
Check out the main repository at https://github.com/matejur/point-tracking-master-thesis.

## Get started

1. Recursively clone this repository

```bash
git clone git@github.com:matejur/dynamic_mast3r.git
```

2. We used the [uv](https://github.com/astral-sh/uv) package manager.
Run the following to download all dependencies.
```bash
uv sync
```

3. Optional, compile the cuda kernels for RoPE.
```bash
cd dust3r/croco/models/curope/
uv run python setup.py build_ext --inplace
cd ../../../../
```

## Checkpoint

Checkpoint from my thesis can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1vlDBwc7zoqEiIJPe_d4fXaMHROsn7pmo?usp=drive_link).

## Training

If you wish to train it yourself you need:
- pretrained MASt3R model from the [original repository](https://github.com/naver/mast3r)
- the PointOdyssey dataset https://pointodyssey.com/
- Panning MOVI-e from the [LocoTrack's repository](https://github.com/cvlab-kaist/locotrack)

Dataloaders require a slighlty different dataset structure, so run `preprocess_kubric.py` and `preprocess_pointodyssey.py`. Check the scripts for setting required folders.

Finally, modify the provided `train.sh` script with necessary paths and run using `uv run bash train.sh`.

Depending on the PyTorch version, you may need to add the `weights_only=False` to the line 140 in `dust3r/dust3r/training.py` to allow loading of the model.

## Evaluation

This model is not evaluated on its own, check the main repository https://github.com/matejur/point-tracking-master-thesis.

## Acknowledgement

Please check out the original [MASt3R repository](https://github.com/naver/mast3r).