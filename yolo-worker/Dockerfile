FROM nvidia/cuda:12.1-devel-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install YOLO dependencies
RUN pip3 install --no-cache-dir \
    ultralytics \
    torch \
    torchvision \
    torchaudio \
    redis \
    pyyaml

# Copy worker script
COPY scripts/worker.py ./scripts/

CMD ["python3", "-u", "./scripts/worker.py"]
