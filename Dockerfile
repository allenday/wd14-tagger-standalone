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

# Create cache directories
RUN mkdir -p /root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs && \
    mkdir -p /root/.cache/huggingface/hub/models--Camais03--camie-tagger/snapshots/latest

# Attempt to set up models, but don't fail the build if it doesn't work
RUN echo "Setting up models..." && \
    python setup_models.py || echo "Setup script failed, but continuing build" && \
    echo "Model setup complete"

# Alternative method to download model files - we're creating separate RUN commands
# to avoid failing the entire build if one step fails
RUN echo "Attempting to download model files..." && \
    python -c "import os; from huggingface_hub import hf_hub_download; \
    os.makedirs('/root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/', exist_ok=True); \
    try: \
        model_path = hf_hub_download('Camais03/camie-tagger', 'model_initial.onnx', local_dir='/root/.cache/huggingface/hub'); \
        print(f'Model downloaded to {model_path}'); \
    except Exception as e: \
        print(f'Model download error: {str(e)}'); \
        print('Will download at runtime'); \
    " || echo "Model download failed, will try at runtime"

# Separate command for tag mapping to avoid failing the entire build
RUN echo "Attempting to download tag mapping..." && \
    python -c "import os; from huggingface_hub import hf_hub_download; \
    os.makedirs('/root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/', exist_ok=True); \
    try: \
        tags_path = hf_hub_download('Camais03/camie-tagger', 'tag_mapping_v1.json', local_dir='/root/.cache/huggingface/hub'); \
        print(f'Tags downloaded to {tags_path}'); \
    except Exception as e: \
        print(f'Tag mapping download error: {str(e)}'); \
        print('Will download at runtime'); \
    " || echo "Tag mapping download failed, will try at runtime"

# Create fallback files if downloads failed
RUN echo "Creating fallback files if needed..." && \
    python -c "import os, json; \
    mapping_file = '/root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/c7dc4e38696a812e593916e3f2e51b92f687f8ea'; \
    if not os.path.exists(mapping_file): \
        print('Creating fallback tag mapping...'); \
        tag_mapping = { \
            'idx_to_tag': {str(i): f'tag_{i}' for i in range(100)}, \
            'tag_to_category': {f'tag_{i}': 'general' for i in range(100)} \
        }; \
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True); \
        with open(mapping_file, 'w') as f: \
            json.dump(tag_mapping, f); \
        print(f'Created fallback tag mapping at {mapping_file}'); \
    " || echo "Fallback creation failed, continuing anyway"

# List what we've got, but don't fail if the directory doesn't exist
RUN ls -la /root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/ || echo "No model files found, will download at runtime"

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