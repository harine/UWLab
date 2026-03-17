# save as inspect_uwlab_obs_images.py and run:
#   python inspect_uwlab_obs_images.py --traj 0

import argparse
from pathlib import Path

import torch
import numpy as np


def _print_array_stats(label: str, arr: np.ndarray) -> None:
    print(f"\n{label}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    if arr.size > 0:
        print(f"  min/max: {arr.min()} / {arr.max()}")
    else:
        print("  (empty array)")


def _print_structure(obj, prefix: str = "data", indent: int = 0) -> None:
    pad = "  " * indent

    if isinstance(obj, dict):
        print(f"{pad}{prefix} (dict, {len(obj)} keys):")
        for k, v in obj.items():
            _print_structure(v, prefix=str(k), indent=indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{prefix} ({type(obj).__name__}, len={len(obj)}):")
        # Just show the first element's structure to avoid huge prints
        if len(obj) > 0:
            _print_structure(obj[0], prefix=f"{prefix}[0]", indent=indent + 1)
    elif isinstance(obj, torch.Tensor):
        arr = obj.numpy()
        print(
            f"{pad}{prefix}: Tensor, shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}"
        )
    elif isinstance(obj, np.ndarray):
        print(
            f"{pad}{prefix}: ndarray, shape={obj.shape}, ndim={obj.ndim}, dtype={obj.dtype}"
        )
    else:
        print(f"{pad}{prefix}: {type(obj).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/harine/UWLab/data/peg_expert_imgs_randomization",
    )
    parser.add_argument(
        "--traj",
        type=int,
        default=0,
        help="trajectory index (e.g. 0 -> traj_000000.pt)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    traj_path = dataset_dir / "trajectories" / f"traj_{args.traj:06d}.pt"
    print(f"Loading {traj_path}")
    data = torch.load(traj_path, map_location="cpu", weights_only=False)

    # ---- full nested structure (keys and dims) ----
    print("\n=== Full data structure (keys and dims) ===")
    _print_structure(data, prefix="data")

    # ---- top-level structure ----
    print("\n=== Top-level keys ===")
    for key, value in data.items():
        vtype = type(value).__name__
        extra = ""
        if isinstance(value, torch.Tensor):
            arr = value.numpy()
            extra = f", shape={arr.shape}, dtype={arr.dtype}"
        elif isinstance(value, np.ndarray):
            extra = f", shape={value.shape}, dtype={value.dtype}"
        elif isinstance(value, dict):
            extra = f", dict_keys={list(value.keys())}"
        print(f"- {key}: {vtype}{extra}")

    # ---- image observations ----
    obs_images = data.get("obs_images", {})
    print(f"\nobs_images keys: {list(obs_images.keys())}")

    for key, value in obs_images.items():
        if isinstance(value, torch.Tensor):
            arr = value.numpy()
        else:
            arr = np.asarray(value)

        print(f"\nKey: {key}")
        print(f"  raw type: {type(value).__name__}")
        _print_array_stats("  data", arr)

    # also check rendered_images if present
    rendered = data.get("rendered_images", None)
    if rendered is not None:
        arr = rendered if isinstance(rendered, np.ndarray) else np.asarray(rendered)
        _print_array_stats("rendered_images", arr)

    # ---- non-image obs dictionaries (for debugging keys) ----
    for group_name in ("obs_proprio", "obs_assets", "obs_other_state"):
        group = data.get(group_name, {})
        if not isinstance(group, dict):
            continue
        print(f"\n{group_name} keys: {list(group.keys())}")
        # show shape/dtype for each key in the group
        for k, v in group.items():
            if isinstance(v, torch.Tensor):
                arr = v.numpy()
            else:
                arr = np.asarray(v)
            _print_array_stats(f"{group_name}[{k}]", arr)


if __name__ == "__main__":
    main()