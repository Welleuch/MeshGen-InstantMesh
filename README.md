# InstantMesh Mesh Generation

Docker-based InstantMesh 3D mesh generation using Zero123++ and InstantMesh.

## Pipeline

1. Input image → rembg → Zero123++ → InstantMesh → STL mesh

**IMPORTANT:** Must apply rembg BEFORE Zero123++, not after! Wrong order causes cube artifacts.

## Quick Start

```bash
# Local test
python run.py configs/instant-mesh-large.yaml input.png --output_path outputs/test --diffusion_steps 28
```

## RunPod Deployment

```bash
# Build
docker build --platform linux/amd64 -t walidelleuch/instantmesh-runpod:latest .
docker push walidelleuch/instantmesh-runpod:latest
```

## Parameters

| Parameter         | Default | Notes                  |
| ----------------- | ------- | ---------------------- |
| grid_res          | 128     | ISO-surface extraction |
| triplane_high_res | 64      | Triplane features      |
| diffusion_steps   | 28      | Zero123++ denoising    |

## VRAM Notes

- RTX 5060 8GB: grid_res=128 works (clean ~42k vertices), >128 causes OOM
- Default settings produce clean mesh without artifacts
