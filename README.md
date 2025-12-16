# Metric-Aligning-Motion-Matching

Computer Graphic course final project. Implementing MAMM base on the original paper.

## File structure

- `original_motion/`: Contain the file for storing original motion that we want to align the control sequence. Stored in `bvh` format.
- `output/`: Contain the aligned motion, also in `bvh` format.

## How to use virtual environment

1. Create an virtual environment via python venv tool

```
python -m venv .venv
```

2. Active the virtual environment

MacOS

```
source ./.venv/bin/activate
```

3. Install all requirements

```
pip install -r requirements.txt
```

## How to watch the bvh motion

Our original motion and aligned motion will be in bvh format, you can visite the online viewer [BVH player](https://lo-th.github.io/olympe/BVH_player.html) to play bvh file.

BVH is a file record the character root movement, joint rotation in every frame.
