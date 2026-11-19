"""Build README poster frames from the released SN001 clips (frames are copied, not synthesized).

The standing poster shows the reset, middle and final frames of the 110k clip. The walking poster
shows the first, middle and last frames of each seed's clip, seed 0 on the top row and seed 2
below. Frames are downscaled by an integer factor only.
"""
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "assets" / "demo"


def frames(path):
    clip = list(iio.imiter(path))
    return [clip[0], clip[len(clip) // 2], clip[-1]]


def row(path, step):
    return np.concatenate([f[::step, ::step] for f in frames(path)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output_dir / "sn001-110k-stand-reach.png", row(DEMO / "sn001-110k-seed-1000-stand-reach.mp4", 2))
    walking = np.concatenate([row(DEMO / "sn001-walking-seed0-200k-forward-lean.mp4", 2),
                              row(DEMO / "sn001-walking-seed2-200k-forward-lean.mp4", 2)], axis=0)
    iio.imwrite(args.output_dir / "sn001-walking-200k-forward-lean.png", walking)


if __name__ == "__main__":
    main()
