FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including CUDA toolkit for nvdiffrast
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-dev \
    libglib2.0-0 \
    cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install nvdiffrast with --no-build-isolation (needs PyTorch already installed)
RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/

# Copy source code
COPY . .

# Download models at runtime (or could be baked in)
# Models are downloaded via huggingface_hub at runtime

EXPOSE 8000

CMD ["python", "runpod_handler.py"]