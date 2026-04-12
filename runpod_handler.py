import os
import torch
import runpod
import trimesh
import rembg
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj
from src.utils.infer_util import remove_background, resize_foreground

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/instant-mesh-large.yaml")
DIFFUSION_MODEL = "sudo-ai/zero123plus-v1.2"
MODEL_REPO = "TencentARC/InstantMesh"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading config...")
config = OmegaConf.load(CONFIG_PATH)
model_config = config.model_config
infer_config = config.infer_config

print("Loading diffusion pipeline...")
pipeline = DiffusionPipeline.from_pretrained(
    DIFFUSION_MODEL,
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)

print("Loading custom white-background UNet...")
try:
    unet_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="diffusion_pytorch_model.bin", repo_type="model"
    )
    state_dict = torch.load(unet_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)
    print("Custom UNet loaded!")
except Exception as e:
    print(f"Warning: Could not load custom UNet: {e}")

pipeline = pipeline.to(device)

print("Loading reconstruction model...")
model = instantiate_from_config(model_config)
try:
    model_ckpt_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="instant_mesh_large.ckpt", repo_type="model"
    )
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")
    }
    model.load_state_dict(state_dict, strict=True)
    print("Model checkpoint loaded!")
except Exception as e:
    print(f"Warning: Could not load model checkpoint: {e}")

model = model.to(device)
model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

print("Initializing rembg session...")
rembg_session = rembg.new_session()


def process_image(input_image: Image.Image) -> Image.Image:
    """Remove background and resize foreground - APPLY BEFORE Zero123++"""
    input_image = remove_background(input_image, rembg_session)
    input_image = resize_foreground(input_image, 0.85)
    return input_image


def generate_multiview(image: Image.Image, diffusion_steps: int = 28) -> torch.Tensor:
    """Generate multiview images from input using CORRECT official reshape"""
    output = pipeline(image, num_inference_steps=diffusion_steps).images[0]

    # Official de-gridify using einops (CORRECT)
    images = np.asarray(output, dtype=np.float32) / 255.0  # (960, 640, 3)
    images = (
        torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    )  # (3, 960, 640)
    images = rearrange(
        images, "c (n h) (m w) -> (n m) c h w", n=3, m=2
    )  # (6, 3, 320, 320)

    return images


def reconstruct_mesh(images: torch.Tensor) -> trimesh.Trimesh:
    """Reconstruct 3D mesh from multiview images."""
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)

    images = images.unsqueeze(0).to(device)

    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)
        mesh_out = model.extract_mesh(planes, **infer_config)

    vertices, faces, vertex_colors = mesh_out

    temp_obj = "/tmp/temp_mesh.obj"
    save_obj(vertices, faces, vertex_colors, temp_obj)

    mesh = trimesh.load(temp_obj)
    os.remove(temp_obj)

    return mesh


def handler(job):
    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    diffusion_steps = job_input.get("diffusion_steps", 28)

    if not image_url:
        return {"error": "No image_url provided"}

    print(f"Processing image: {image_url}")
    print(f"Diffusion steps: {diffusion_steps}")

    response = requests.get(image_url)
    input_image = Image.open(BytesIO(response.content)).convert("RGB")

    # Stage 1: Remove background BEFORE Zero123++ (CORRECT pipeline order)
    print("Removing background...")
    input_image = process_image(input_image)

    # Stage 2: Generate multiview
    print("Generating multiview...")
    multiview_images = generate_multiview(input_image, diffusion_steps)

    # Stage 3: Reconstruct mesh
    print("Reconstructing mesh...")
    mesh = reconstruct_mesh(multiview_images)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    stl_path = "/tmp/result.stl"
    mesh.export(stl_path)

    with open(stl_path, "rb") as f:
        mesh_bytes = f.read()

    return {
        "status": "success",
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
