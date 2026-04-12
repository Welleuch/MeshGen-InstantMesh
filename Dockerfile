FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Set CUDA_HOME environment variable (already set in nvidia/cuda image, but ensure it)
ENV CUDA_HOME=/usr/local/cuda

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install nvdiffrast (needs PyTorch and CUDA already installed)
RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/

# Copy source code
COPY . .

EXPOSE 8000

CMD ["python", "runpod_handler.py"]