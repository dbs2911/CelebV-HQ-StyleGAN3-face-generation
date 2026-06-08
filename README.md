# CelebV-HQ Face Generation — Inference Code (StyleGAN3-T)

Inference code to reproduce our final submission for the CelebV-HQ
$256\times256$ face-generation challenge.

**Final model.** A StyleGAN3-T generator (transfer-learned from NVIDIA
`ffhqu-256`, fine-tuned on a cleaned CelebV-HQ dataset), with two
inference-time refinements:

1. **Checkpoint blending** — average the `G_ema` weights of three strong
   training snapshots (kimg 1300 / 1600 / 1900) into one generator.
2. **Lightly relaxed truncation** — truncation `psi = 1.1`.

**Final scores:** FID 32.43 · IS 4.59 · KID 0.0058 · TopPR 0.840.

**Mandatory seeds.** The 1000 submitted images use **seeds 0–999**
(`img_0000.png` ← seed 0, …, `img_0999.png` ← seed 999), `noise_mode=const`,
`psi=1.1`. The exact mapping is in [`submission_seeds.txt`](submission_seeds.txt).

> Generation is fully deterministic: the same seed always yields the same image
> (verified pixel-for-pixel across independent re-runs).

---

## 1. Setup

Requires a CUDA GPU + toolkit and a C++ compiler (StyleGAN3 builds custom CUDA
ops on first run). Verified on Python 3.10, CUDA 12.8, RTX 4090 (24 GB).

```bash
# (a) clone NVIDIA StyleGAN3 at the exact pinned commit
git clone https://github.com/NVlabs/stylegan3 external/stylegan3
git -C external/stylegan3 checkout c233a919a6faee6e36a316ddd4eddababad1adf9

# (b) PyTorch (CUDA-matched wheel; NOT from PyPI)
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision

# (c) remaining inference deps
pip install -r requirements.txt
```

## 2. Checkpoints

Place the three fine-tuned StyleGAN3-T snapshots under `checkpoints/`:

```
checkpoints/network-snapshot-001300.pkl
checkpoints/network-snapshot-001600.pkl
checkpoints/network-snapshot-001900.pkl
```

> Weights are **not** included in this repository (challenge rules: top-10
> teams submit weights separately). They are provided with the weight package.

## 3. Reproduce the final 1000 images

```bash
# (a) build the blended generator
python blend_ckpts.py \
    --ckpts checkpoints/network-snapshot-001300.pkl \
            checkpoints/network-snapshot-001600.pkl \
            checkpoints/network-snapshot-001900.pkl \
    --out   checkpoints/blended_v10_1300_1600_1900.pkl

# (b) generate the 1000 submitted images (seeds 0..999, psi=1.1)
python generate_images.py \
    --network   checkpoints/blended_v10_1300_1600_1900.pkl \
    --outdir    out/submission \
    --count     1000 \
    --trunc     1.1 \
    --seed0     0 \
    --noise-mode const
```

This writes `out/submission/img_0000.png … img_0999.png` and a `seeds.txt`
identical (in header and mapping) to [`submission_seeds.txt`](submission_seeds.txt).

## Files

| file | purpose |
|------|---------|
| `blend_ckpts.py`        | average `G_ema` of N checkpoints into one generator |
| `generate_images.py`    | sample N images from a generator with recorded seeds |
| `submission_seeds.txt`  | seed → filename mapping for the final 1000 images |
| `requirements.txt`      | inference dependencies (PyTorch installed separately) |
| `Dockerfile`            | optional reproducible CUDA environment |

## Academic integrity

All training images derive from the provided CelebV-HQ source. No
internet-scraped real images and no outputs of external generation services
(e.g. Midjourney, DALL·E) are used. Generation is from our own trained weights
only.
