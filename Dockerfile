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
RUN pip install --no-cache-dir google-cloud-storage==2.16.0

# Copy function code
COPY main.py vector_generator.py ./
COPY tagger/ ./tagger/

# Copy setup script
COPY setup_models.py /workspace/

# Download models during build time to avoid cold start downloads
RUN echo "Setting up models..." && \
    python setup_models.py && \
    echo "Model setup complete"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FUNCTION_TARGET=process_pubsub_message
ENV FUNCTION_SIGNATURE_TYPE=cloudevent

# Cloud Function entry point
CMD ["functions-framework", "--target=${FUNCTION_TARGET}", "--signature-type=${FUNCTION_SIGNATURE_TYPE}"]

# Expose port for functions-framework
EXPOSE 8080