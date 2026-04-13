import os
import sys
import torch
import runpod
import trimesh
import rembg
import requests
import traceback
import numpy as np
from PIL import Image
from io import BytesIO
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/instant-mesh-large.yaml")
DIFFUSION_MODEL = "sudo-ai/zero123plus-v1.2"
MODEL_REPO = "TencentARC/InstantMesh"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Using device: {device}", flush=True)
print(f"[INIT] PyTorch version: {torch.__version__}", flush=True)
print(f"[INIT] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[INIT] CUDA version: {torch.version.cuda}", flush=True)
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"[INIT] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        flush=True,
    )

pipeline = None
model = None
rembg_session = None


def initialize_models():
    """Lazy initialization - only loads models when first job arrives"""
    global pipeline, model, rembg_session

    print("[INIT] Starting model loading...", flush=True)

    try:
        print("[INIT] Loading config...")
        config = OmegaConf.load(CONFIG_PATH)
        model_config = config.model_config
        infer_config = config.infer_config
        print(
            f"[INIT] Config loaded: grid_res={model_config.get('grid_res')}", flush=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", flush=True)
        traceback.print_exc()
        raise

    try:
        print("[INIT] Loading diffusion pipeline (this may take a few minutes)...")
        pipeline = DiffusionPipeline.from_pretrained(
            DIFFUSION_MODEL,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        print("[INIT] Loading custom white-background UNet...")
        try:
            unet_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="diffusion_pytorch_model.bin",
                repo_type="model",
            )
            state_dict = torch.load(unet_path, map_location="cpu")
            pipeline.unet.load_state_dict(state_dict, strict=True)
            print("[INIT] Custom UNet loaded!", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not load custom UNet: {e}", flush=True)

        # Enable CPU offloading to save VRAM
        print("[INIT] Enabling model CPU offloading for VRAM savings...", flush=True)
        pipeline.enable_model_cpu_offload()

        print("[INIT] Diffusion pipeline loaded successfully!", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load diffusion pipeline: {e}", flush=True)
        traceback.print_exc()
        raise

    try:
        print("[INIT] Loading reconstruction model...")
        from src.utils.train_util import instantiate_from_config

        model = instantiate_from_config(model_config)

        print("[INIT] Downloading model checkpoint...")
        model_ckpt_path = hf_hub_download(
            repo_id=MODEL_REPO, filename="instant_mesh_large.ckpt", repo_type="model"
        )
        state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")
        }
        model.load_state_dict(state_dict, strict=True)

        # Keep model on CPU initially - move to GPU only when needed
        # model = model.to(device)
        model = model.to("cpu")
        model.init_flexicubes_geometry(device, fovy=30.0)
        model = model.eval()
        print("[INIT] Reconstruction model loaded successfully!", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load reconstruction model: {e}", flush=True)
        traceback.print_exc()
        raise

    try:
        print("[INIT] Initializing rembg session...")
        rembg_session = rembg.new_session()
        print("[INIT] rembg session initialized!", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to initialize rembg: {e}", flush=True)
        traceback.print_exc()
        raise

    print("[INIT] All models loaded successfully!", flush=True)


def process_image(input_image: Image.Image) -> Image.Image:
    """Remove background and resize foreground - APPLY BEFORE Zero123++"""
    from src.utils.infer_util import remove_background, resize_foreground

    input_image = remove_background(input_image, rembg_session)
    input_image = resize_foreground(input_image, 0.85)
    return input_image


def generate_multiview(image: Image.Image, diffusion_steps: int = 28) -> torch.Tensor:
    """Generate multiview images from input using CORRECT official reshape"""
    output = pipeline(image, num_inference_steps=diffusion_steps).images[0]

    images = np.asarray(output, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)

    return images


def reconstruct_mesh(images: torch.Tensor) -> trimesh.Trimesh:
    """Reconstruct 3D mesh from multiview images."""
    from src.utils.camera_util import get_zero123plus_input_cameras
    from src.utils.mesh_util import save_obj

    # Move model to GPU for inference
    model_gpu = model.to(device)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    images = images.unsqueeze(0).to(device)

    with torch.no_grad():
        planes = model_gpu.forward_planes(images, input_cameras)
        mesh_out = model_gpu.extract_mesh(planes, **infer_config)

    # Move model back to CPU to free VRAM
    model_gpu = model_gpu.to("cpu")
    torch.cuda.empty_cache()

    vertices, faces, vertex_colors = mesh_out

    temp_obj = "/tmp/temp_mesh.obj"
    save_obj(vertices, faces, vertex_colors, temp_obj)

    mesh = trimesh.load(temp_obj)
    os.remove(temp_obj)

    return mesh


def handler(job):
    global pipeline, model, rembg_session, infer_config

    # Lazy initialization - load models on first job
    if pipeline is None or model is None:
        try:
            initialize_models()
            # Load config again for infer_config
            config = OmegaConf.load(CONFIG_PATH)
            infer_config = config.infer_config
        except Exception as e:
            return {
                "error": f"Initialization failed: {str(e)}",
                "trace": traceback.format_exc(),
            }

    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    diffusion_steps = job_input.get("diffusion_steps", 28)

    if not image_url:
        return {"error": "No image_url provided"}

    print(f"[JOB] Processing image: {image_url}", flush=True)
    print(f"[JOB] Diffusion steps: {diffusion_steps}", flush=True)

    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        input_image = Image.open(BytesIO(response.content)).convert("RGB")

        # Stage 1: Remove background BEFORE Zero123++ (CORRECT pipeline order)
        print("[JOB] Removing background...", flush=True)
        input_image = process_image(input_image)

        # Stage 2: Generate multiview
        print("[JOB] Generating multiview...", flush=True)
        multiview_images = generate_multiview(input_image, diffusion_steps)
        print(f"[JOB] Multiview shape: {multiview_images.shape}", flush=True)

        # Stage 3: Reconstruct mesh
        print("[JOB] Reconstructing mesh...", flush=True)
        mesh = reconstruct_mesh(multiview_images)

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        stl_path = "/tmp/result.stl"
        mesh.export(stl_path)

        with open(stl_path, "rb") as f:
            mesh_bytes = f.read()

        print(
            f"[JOB] Success! Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}",
            flush=True,
        )

        return {
            "status": "success",
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "stl_size_bytes": len(mesh_bytes),
        }

    except Exception as e:
        print(f"[ERROR] Job failed: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    print("[MAIN] Starting RunPod serverless handler...", flush=True)
    runpod.serverless.start({"handler": handler})
