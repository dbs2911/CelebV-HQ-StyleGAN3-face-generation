"""Sample N images from a trained StyleGAN3 pkl and record the seeds.

Writes:
  <outdir>/img_0000.png ... img_<N-1>.png
  <outdir>/seeds.txt    — one "img_XXXX.png\\t<seed>" line per image.

Seed recording is required: the organizers' top-10 verification re-runs
the inference code with the declared seeds and compares to the submitted
images pixel-for-pixel (mismatch = 0 points).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch

# Make the cloned stylegan3 importable.
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


def load_generator(pkl_path: Path, device: torch.device):
    with dnnlib.util.open_url(str(pkl_path)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()
    return G


@torch.no_grad()
def sample_one(G, seed: int, trunc: float, device: torch.device, noise_mode: str = "const") -> np.ndarray:
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    if noise_mode == "random":
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    img = G(z, label, truncation_psi=trunc, noise_mode=noise_mode)
    img = (img.clamp(-1, 1) + 1) * 127.5
    img = img.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()[0]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", type=Path, required=True, help="Path to generator .pkl")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--trunc", type=float, default=0.7)
    ap.add_argument("--seed0", type=int, default=0, help="First seed; seeds are seed0..seed0+count-1")
    ap.add_argument("--fmt", choices=["png", "jpg"], default="png")
    ap.add_argument("--noise-mode", dest="noise_mode", choices=["const", "random", "none"], default="const",
                    help="StyleGAN3 generator noise mode (const = deterministic per-layer noise, default)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = load_generator(args.network, device)

    seeds = list(range(args.seed0, args.seed0 + args.count))
    seeds_log = args.outdir / "seeds.txt"
    with open(seeds_log, "w") as fout:
        fout.write(f"# network={args.network.name} trunc={args.trunc} noise_mode={args.noise_mode}\n")
        for i, seed in enumerate(seeds):
            img = sample_one(G, seed, args.trunc, device, args.noise_mode)
            name = f"img_{i:04d}.{args.fmt}"
            out_path = args.outdir / name
            if args.fmt == "png":
                PIL.Image.fromarray(img).save(out_path, optimize=True)
            else:
                PIL.Image.fromarray(img).save(out_path, quality=95)
            fout.write(f"{name}\t{seed}\n")
            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{args.count}]", flush=True)

    total = sum(f.stat().st_size for f in args.outdir.glob(f"img_*.{args.fmt}"))
    print(f"[done] {args.count} images -> {args.outdir} ({total / 1e6:.1f} MB)")
    print(f"[seeds] {seeds_log}")


if __name__ == "__main__":
    main()
