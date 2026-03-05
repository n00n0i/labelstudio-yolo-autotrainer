FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Docker CLI (for running training containers)
RUN curl -fsSL https://get.docker.com | sh

# Copy application
COPY webhook_handler.py .

EXPOSE 8000

CMD ["python3", "-u", "webhook_handler.py"]
