"""Average G_ema weights across multiple StyleGAN3 ckpts and save a new pkl.

Use case (v10 sub_024 attempt): combine 3 ckpts near v10 best (kimg 1300/
1600/1900, all in 8.x fid bucket) into one G_ema. Hypothesis is that the
averaged weights live in a smoother basin of the loss surface, so the
generated distribution is less mode-flickery than any individual ckpt.

This is a generation-time-only trick: it does not retrain, does not change
the architecture, and reproduces deterministically from a fixed seed list,
so it stays within the leaderboard rules.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SG3 = ROOT / "external" / "stylegan3"
if not SG3.exists():
    raise SystemExit(
        f"stylegan3 repo missing at {SG3}. Run:\n"
        f"  git clone https://github.com/NVlabs/stylegan3 {SG3}\n"
        f"  git -C {SG3} checkout c233a919a6faee6e36a316ddd4eddababad1adf9"
    )
sys.path.insert(0, str(SG3))

import dnnlib  # noqa: E402
import legacy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="ckpt pkls to average (G_ema)")
    ap.add_argument("--out", type=Path, required=True, help="output pkl path")
    args = ap.parse_args()

    if len(args.ckpts) < 2:
        raise SystemExit("Need at least 2 ckpts to blend")

    print(f"[blend] averaging G_ema from {len(args.ckpts)} ckpts:")
    for c in args.ckpts:
        print(f"  - {c}")

    # Load the first ckpt fully (so we can re-save the same dict structure
    # with the averaged G_ema swapped in).
    with dnnlib.util.open_url(args.ckpts[0]) as f:
        base = legacy.load_network_pkl(f)

    # Collect G_ema state_dicts.
    state_dicts = []
    for c in args.ckpts:
        with dnnlib.util.open_url(c) as f:
            data = legacy.load_network_pkl(f)
        state_dicts.append({k: v.detach().clone() for k, v in data["G_ema"].state_dict().items()})

    # Verify same keys.
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != keys:
            raise SystemExit("ckpts have mismatched parameter keys")

    # Average.
    avg = {}
    for k in keys:
        tensors = [sd[k] for sd in state_dicts]
        if tensors[0].is_floating_point():
            stacked = torch.stack([t.float() for t in tensors], dim=0)
            avg[k] = stacked.mean(dim=0).to(tensors[0].dtype)
        else:
            # ints / buffers — keep first (e.g., augment counters).
            avg[k] = tensors[0]

    # Load averaged weights into base G_ema model.
    missing, unexpected = base["G_ema"].load_state_dict(avg, strict=True)
    print(f"[blend] loaded averaged state into G_ema (missing={missing}, unexpected={unexpected})")

    # Also blend G and D? Match common practice: only G_ema is used for
    # inference. Leave G and D as-is from base for reproducibility safety.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        import pickle
        pickle.dump(base, f)
    print(f"[done] wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
