FROM python:3.10-slim

# Cloud Functions use /workspace as the default working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy function code
COPY main.py vector_generator.py ./
COPY tagger/ ./tagger/

# Download models during build time to avoid cold start downloads
RUN mkdir -p /root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/ /root/.cache/huggingface/hub/models--Camais03--camie-tagger/snapshots/latest/ \
    && echo "Downloading model files..." \
    && python -c "\
    import os\
    \nimport json\
    \nimport requests\
    \nfrom huggingface_hub import hf_hub_download\
    \n\
    \ntry:\
    \n    print('Downloading ONNX model...')\
    \n    model_path = hf_hub_download(repo_id='Camais03/camie-tagger', filename='model_initial.onnx', cache_dir='/root/.cache/huggingface')\
    \n    print(f'Downloaded model to {model_path}')\
    \nexcept Exception as e:\
    \n    print(f'Error downloading model: {str(e)}')\
    \n\
    \ntry:\
    \n    print('Creating fallback tag mapping...')\
    \n    # Create a simple fallback mapping if download fails\
    \n    tag_mapping = {\
    \n        'idx_to_tag': {str(i): f'tag_{i}' for i in range(100)},\
    \n        'tag_to_category': {f'tag_{i}': 'general' for i in range(100)}\
    \n    }\
    \n    \
    \n    mapping_file = '/root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/c7dc4e38696a812e593916e3f2e51b92f687f8ea'\
    \n    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)\
    \n    with open(mapping_file, 'w') as f:\
    \n        json.dump(tag_mapping, f)\
    \n    print(f'Created fallback mapping at {mapping_file}')\
    \nexcept Exception as e:\
    \n    print(f'Error creating fallback mapping: {str(e)}')\
    "\
    && echo "Model setup complete"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FUNCTION_TARGET=process_pubsub_message
ENV FUNCTION_SIGNATURE_TYPE=cloudevent

# Cloud Function entry point
CMD ["functions-framework", "--target=${FUNCTION_TARGET}", "--signature-type=${FUNCTION_SIGNATURE_TYPE}"]

# Expose port for functions-framework
EXPOSE 8080