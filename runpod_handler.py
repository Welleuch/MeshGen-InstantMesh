#!/usr/bin/env python3
"""
RunPod handler for InstantMesh 3D generation.
Enhanced with comprehensive debugging and logging.
"""

import os
import sys
import gc
import torch
import runpod
import trimesh
import rembg
import requests
import traceback
import numpy as np
from PIL import Image
from io import BytesIO

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/instant-mesh-large.yaml")
DIFFUSION_MODEL = "sudo-ai/zero123plus-v1.2"
MODEL_REPO = "TencentARC/InstantMesh"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg, level="INFO"):
    """Enhanced logging with timestamp and flush"""
    import time

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)
    sys.stderr.flush()
    sys.stdout.flush()


log(f"=== STARTING HANDLER ===", "INFO")
log(f"Python version: {sys.version}", "INFO")
log(f"Using device: {device}", "INFO")

if torch.cuda.is_available():
    log(f"CUDA available: True", "INFO")
    log(f"CUDA version: {torch.version.cuda}", "INFO")
    log(f"GPU: {torch.cuda.get_device_name(0)}", "INFO")
    props = torch.cuda.get_device_properties(0)
    log(f"GPU memory: {props.total_memory / 1e9:.2f} GB", "INFO")
else:
    log("CUDA available: False", "WARNING")

pipeline = None
model = None
rembg_session = None
infer_config = None


def get_memory_info():
    """Get current memory usage"""
    info = {}
    if torch.cuda.is_available():
        info["gpu_allocated"] = torch.cuda.memory_allocated() / 1e9
        info["gpu_reserved"] = torch.cuda.memory_reserved() / 1e9
        info["gpu_free"] = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()
        ) / 1e9
    return info


def initialize_models():
    """Lazy initialization with detailed memory tracking"""
    global pipeline, model, rembg_session, infer_config

    log("=== STARTING MODEL INITIALIZATION ===", "INFO")
    log(f"Memory before init: {get_memory_info()}", "INFO")

    try:
        log("Loading config...", "INFO")
        config = OmegaConf.load(CONFIG_PATH)
        model_config = config.model_config
        infer_config = config.infer_config
        log(f"Config loaded: grid_res={model_config.get('grid_res')}", "INFO")
        log(f"Memory after config: {get_memory_info()}", "INFO")
    except Exception as e:
        log(f"ERROR: Failed to load config: {e}", "ERROR")
        traceback.print_exc()
        sys.stderr.flush()
        raise

    try:
        log("Loading diffusion pipeline...", "INFO")
        log(f"Memory before pipeline: {get_memory_info()}", "INFO")

        pipeline = DiffusionPipeline.from_pretrained(
            DIFFUSION_MODEL,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        log("Loading custom white-background UNet...", "INFO")
        try:
            unet_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="diffusion_pytorch_model.bin",
                repo_type="model",
            )
            state_dict = torch.load(unet_path, map_location="cpu")
            pipeline.unet.load_state_dict(state_dict, strict=True)
            log("Custom UNet loaded!", "INFO")
        except Exception as e:
            log(f"WARNING: Could not load custom UNet: {e}", "WARNING")

        # Don't move to GPU yet - keep on CPU until inference
        log("Pipeline loaded (keeping on CPU)", "INFO")
        log(f"Memory after pipeline: {get_memory_info()}", "INFO")

    except Exception as e:
        log(f"ERROR: Failed to load diffusion pipeline: {e}", "ERROR")
        traceback.print_exc()
        sys.stderr.flush()
        raise

    try:
        log("Loading reconstruction model...", "INFO")
        log(f"Memory before model: {get_memory_info()}", "INFO")

        from src.utils.train_util import instantiate_from_config

        model = instantiate_from_config(model_config)

        log("Downloading model checkpoint...", "INFO")
        model_ckpt_path = hf_hub_download(
            repo_id=MODEL_REPO, filename="instant_mesh_large.ckpt", repo_type="model"
        )
        state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")
        }
        model.load_state_dict(state_dict, strict=True)

        model = model.to("cpu")
        model.init_flexicubes_geometry(device, fovy=30.0)
        model = model.eval()

        log("Reconstruction model loaded!", "INFO")
        log(f"Memory after model: {get_memory_info()}", "INFO")

    except Exception as e:
        log(f"ERROR: Failed to load reconstruction model: {e}", "ERROR")
        traceback.print_exc()
        sys.stderr.flush()
        raise

    try:
        log("Initializing rembg session...", "INFO")
        rembg_session = rembg.new_session()
        log("rembg session initialized!", "INFO")
    except Exception as e:
        log(f"ERROR: Failed to initialize rembg: {e}", "ERROR")
        traceback.print_exc()
        sys.stderr.flush()
        raise

    log("=== ALL MODELS LOADED SUCCESSFULLY ===", "INFO")
    log(f"Final memory: {get_memory_info()}", "INFO")


def process_image(input_image: Image.Image) -> Image.Image:
    """Remove background and resize foreground"""
    from src.utils.infer_util import remove_background, resize_foreground

    log("Processing image (rembg)...", "INFO")
    input_image = remove_background(input_image, rembg_session)
    input_image = resize_foreground(input_image, 0.85)
    return input_image


def generate_multiview(image: Image.Image, diffusion_steps: int = 28) -> torch.Tensor:
    """Generate multiview images"""
    log(f"Generating multiview (steps={diffusion_steps})...", "INFO")
    log(f"Memory before pipeline: {get_memory_info()}", "INFO")

    # Move pipeline to GPU for inference
    pipeline_gpu = pipeline.to(device)

    output = pipeline_gpu(image, num_inference_steps=diffusion_steps).images[0]

    # Move back to CPU to free VRAM
    pipeline_gpu = pipeline_gpu.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    log(f"Memory after pipeline: {get_memory_info()}", "INFO")

    images = np.asarray(output, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)

    log(f"Multiview shape: {images.shape}", "INFO")
    return images


def reconstruct_mesh(images: torch.Tensor) -> trimesh.Trimesh:
    """Reconstruct 3D mesh from multiview images"""
    from src.utils.camera_util import get_zero123plus_input_cameras
    from src.utils.mesh_util import save_obj

    log("Reconstructing mesh...", "INFO")
    log(f"Memory before model: {get_memory_info()}", "INFO")

    # Move model to GPU for inference
    model_gpu = model.to(device)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    images_gpu = images.unsqueeze(0).to(device)

    with torch.no_grad():
        planes = model_gpu.forward_planes(images_gpu, input_cameras)
        mesh_out = model_gpu.extract_mesh(planes, **infer_config)

    # Move model back to CPU to free VRAM
    model_gpu = model_gpu.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    log(f"Memory after model: {get_memory_info()}", "INFO")

    vertices, faces, vertex_colors = mesh_out

    temp_obj = "/tmp/temp_mesh.obj"
    save_obj(vertices, faces, vertex_colors, temp_obj)

    mesh = trimesh.load(temp_obj)
    os.remove(temp_obj)

    return mesh


def handler(job):
    global pipeline, model, rembg_session, infer_config

    log("=== NEW JOB RECEIVED ===", "INFO")

    # Lazy initialization
    if pipeline is None or model is None:
        try:
            initialize_models()
        except Exception as e:
            log(f"CRITICAL: Initialization failed: {e}", "ERROR")
            return {
                "error": f"Initialization failed: {str(e)}",
                "trace": traceback.format_exc(),
            }

    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    diffusion_steps = job_input.get("diffusion_steps", 28)

    if not image_url:
        return {"error": "No image_url provided"}

    log(f"Processing: {image_url}", "INFO")

    try:
        # Download image
        log("Downloading image...", "INFO")
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        input_image = Image.open(BytesIO(response.content)).convert("RGB")
        log(f"Image size: {input_image.size}", "INFO")

        # Stage 1: Remove background
        log("Stage 1: Remove background...", "INFO")
        input_image = process_image(input_image)

        # Stage 2: Generate multiview
        log("Stage 2: Generate multiview...", "INFO")
        multiview_images = generate_multiview(input_image, diffusion_steps)

        # Stage 3: Reconstruct mesh
        log("Stage 3: Reconstruct mesh...", "INFO")
        mesh = reconstruct_mesh(multiview_images)

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Export to STL
        stl_path = "/tmp/result.stl"
        mesh.export(stl_path)

        with open(stl_path, "rb") as f:
            mesh_bytes = f.read()

        log(
            f"SUCCESS! Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}", "INFO"
        )
        log(f"STL size: {len(mesh_bytes)} bytes", "INFO")
        log(f"Final memory: {get_memory_info()}", "INFO")

        return {
            "status": "success",
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "stl_size_bytes": len(mesh_bytes),
        }

    except Exception as e:
        log(f"ERROR: Job failed: {e}", "ERROR")
        traceback.print_exc()
        sys.stderr.flush()
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    log("Starting RunPod serverless handler...", "INFO")
    sys.stdout.flush()
    sys.stderr.flush()

    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        log(f"CRITICAL: Server start failed: {e}", "ERROR")
        traceback.print_exc()
        sys.stderr.flush()
        raise
