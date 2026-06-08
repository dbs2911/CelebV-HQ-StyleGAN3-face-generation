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
> (verified pixel-for-pixel across independent re-runs, including a clean
> `docker build --no-cache` reproduction).

---

## 1. Get the weights

The model weights are **not** included in this repository (challenge rules:
top-10 teams submit weights separately). Put the three fine-tuned StyleGAN3-T
snapshots from the weight package into a folder, e.g. `weights/`:

```
weights/network-snapshot-001300.pkl
weights/network-snapshot-001600.pkl
weights/network-snapshot-001900.pkl
```

## 2. Reproduce — Option A: Docker (recommended)

The Docker image bundles the full environment (CUDA 12.8, the exact PyTorch
build, all dependencies, and the pinned StyleGAN3 source with the required
compatibility fix). Requires the NVIDIA Container Toolkit (`--gpus`).

```bash
# (1) build the environment — clones/patches StyleGAN3 and installs everything
docker build -t facegen .

# (2) reproduce the 1000 images (mount the weights folder + an output folder)
docker run --rm --gpus all \
    -v "$(pwd)/weights":/workspace/checkpoints \
    -v "$(pwd)/out":/workspace/out \
    facegen bash -lc '
      python blend_ckpts.py \
        --ckpts checkpoints/network-snapshot-001300.pkl \
                checkpoints/network-snapshot-001600.pkl \
                checkpoints/network-snapshot-001900.pkl \
        --out out/blended_v10_1300_1600_1900.pkl
      python generate_images.py \
        --network out/blended_v10_1300_1600_1900.pkl \
        --outdir out/submission --count 1000 \
        --trunc 1.1 --seed0 0 --noise-mode const
    '
```

The 1000 images appear on the host at `out/submission/img_0000.png … img_0999.png`,
together with a `seeds.txt` whose header and mapping match
[`submission_seeds.txt`](submission_seeds.txt). (StyleGAN3 compiles its custom
CUDA ops on the first `generate` call inside the container; this is normal and
takes a minute.)

## 3. Reproduce — Option B: Manual (no Docker)

Requires a CUDA GPU + toolkit and a C++ compiler. Verified on Python 3.10,
CUDA 12.8, PyTorch 2.9.1, RTX 4090 (24 GB).

```bash
# (a) clone StyleGAN3 at the pinned commit + inline compatibility fix for newer
#     PyTorch (without it, generation fails with "No module named
#     'bias_act_plugin'") — this is exactly what the Dockerfile automates.
git clone https://github.com/NVlabs/stylegan3 external/stylegan3
git -C external/stylegan3 checkout c233a919a6faee6e36a316ddd4eddababad1adf9
sed -i \
  -e 's/^\( *\)torch\.utils\.cpp_extension\.load(name=module_name,/\1module = torch.utils.cpp_extension.load(name=module_name,/' \
  -e '/module = importlib\.import_module(module_name)/d' \
  external/stylegan3/torch_utils/custom_ops.py

# (b) PyTorch (CUDA-matched wheel; NOT from PyPI) + remaining deps
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision
pip install -r requirements.txt

# (c) blend + generate (weights in ./weights/)
python blend_ckpts.py \
    --ckpts weights/network-snapshot-001300.pkl \
            weights/network-snapshot-001600.pkl \
            weights/network-snapshot-001900.pkl \
    --out   weights/blended_v10_1300_1600_1900.pkl
python generate_images.py \
    --network weights/blended_v10_1300_1600_1900.pkl \
    --outdir  out/submission --count 1000 \
    --trunc 1.1 --seed0 0 --noise-mode const
```

## Files

| file | purpose |
|------|---------|
| `Dockerfile`            | reproducible CUDA environment (recommended path) |
| `blend_ckpts.py`        | average `G_ema` of N checkpoints into one generator |
| `generate_images.py`    | sample N images from a generator with recorded seeds |
| `submission_seeds.txt`  | seed → filename mapping for the final 1000 images |
| `requirements.txt`      | inference dependencies (PyTorch installed separately) |

## Academic integrity

All training images derive from the provided CelebV-HQ source. No
internet-scraped real images and no outputs of external generation services
(e.g. Midjourney, DALL·E) are used. Generation is from our own trained weights
only.
