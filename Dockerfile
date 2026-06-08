# Reproducible CUDA environment for CelebV-HQ StyleGAN3 inference.
# StyleGAN3 compiles custom CUDA ops at runtime, so we need the CUDA *devel*
# image (nvcc + headers), not just runtime.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev git build-essential ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch (CUDA 12.8 matched) + inference deps
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pinned StyleGAN3 source
RUN git clone https://github.com/NVlabs/stylegan3 external/stylegan3 \
    && git -C external/stylegan3 checkout c233a919a6faee6e36a316ddd4eddababad1adf9

COPY . .

# Mount checkpoints/ at runtime, then follow README step 3.
CMD ["bash"]
