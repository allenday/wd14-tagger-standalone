FROM python:3.10-slim

# Cloud Functions use /workspace as the default working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt
RUN pip install --no-cache-dir google-cloud-storage==2.16.0

# Copy function code
COPY main.py vector_generator.py ./
COPY tagger/ ./tagger/

# Copy setup script
COPY setup_models.py /workspace/

# Download models during build time to avoid cold start downloads
# This ensures the model files are cached in the Docker image
# Set environment variable to prevent model loading during build
ENV IN_DOCKER_BUILD=1

# IMPORTANT: Before building the Docker image, run:
# python scripts/model_cache.py --output-dir hf_cache --force
# This will download and cache the model files in the correct structure
# The --force flag ensures any existing cache is rebuilt from scratch

# Create huggingface cache directory
RUN mkdir -p /root/.cache/huggingface/hub

# Copy the entire huggingface cache structure that was prepared by model_cache.py
# This includes model files in the exact structure that huggingface_hub expects
COPY hf_cache/hub /root/.cache/huggingface/hub

# Better debugging for cache structure
RUN echo "Checking HuggingFace cache structure..." && \
    find /root/.cache/huggingface -type d | sort && \
    echo "Files in models directory:" && \
    find /root/.cache/huggingface/hub -type f | sort

# Verify the model files are in place
RUN ls -la /root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/ || echo "Model files not found in blobs directory!"
RUN ls -la /root/.cache/huggingface/hub/models--Camais03--camie-tagger/snapshots/latest/ || echo "Snapshot files not found in latest directory!"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FUNCTION_TARGET=process_pubsub_message
ENV FUNCTION_SIGNATURE_TYPE=cloudevent

# Reset the build flag for runtime
ENV IN_DOCKER_BUILD=0

# Cloud Function entry point
CMD ["functions-framework", "--target=process_pubsub_message", "--signature-type=cloudevent"]

# Expose port for functions-framework
EXPOSE 8080