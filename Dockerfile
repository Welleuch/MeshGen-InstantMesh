FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ensure CUDA_HOME is set
ENV CUDA_HOME=/usr/local/cuda

# Copy requirements (torch/torchaudio/torchvision already pre-installed in base image)
COPY requirements.txt .

# Install Python dependencies (excluding torch which is pre-installed)
RUN pip install --no-cache-dir -r requirements.txt

# Install nvdiffrast (needs PyTorch and CUDA already installed)
RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/

# Copy source code
COPY . .

EXPOSE 8000

CMD ["python", "runpod_handler.py"]