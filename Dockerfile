# 1. Corrected Tag
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 2. System dependencies for Git, C++ compilation, and 3D Rendering
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Environment variables for compilation
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA="1"

# 4. Install requirements (assuming you have a requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Install nvdiffrast specifically with no-build-isolation
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

COPY . .

# RunPod Specifics (adjust if your entrypoint file is different)
CMD ["python", "runpod_handler.py"]