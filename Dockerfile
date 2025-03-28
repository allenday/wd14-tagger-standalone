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
RUN mkdir -p /root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/ \
    && python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='Camais03/camie-tagger', filename='model_initial.onnx', cache_dir='/root/.cache/huggingface'); \
    hf_hub_download(repo_id='Camais03/camie-tagger', filename='tag_mapping_v1.json', local_dir='/root/.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FUNCTION_TARGET=process_pubsub_message
ENV FUNCTION_SIGNATURE_TYPE=cloudevent

# Cloud Function entry point
CMD ["functions-framework", "--target=${FUNCTION_TARGET}", "--signature-type=${FUNCTION_SIGNATURE_TYPE}"]

# Expose port for functions-framework
EXPOSE 8080