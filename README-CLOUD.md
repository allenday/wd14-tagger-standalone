# Image Tagger Cloud Function

This Google Cloud Function receives images via Pub/Sub, analyzes them using various image tagging models, and stores both the images and vector representations for search.

## Features

- Receives images from Pub/Sub messages
- Stores original images in Google Cloud Storage
- Analyzes images using ML models (Camie Tagger, etc.)
- Generates sparse vector representations
- Stores vectors in Qdrant for semantic search
- Scale-to-zero for cost efficiency

## Architecture

```
[Image Source] → [Pub/Sub] → [Cloud Function] → [GCS + Qdrant]
```

## Deployment

### Prerequisites

1. Google Cloud Project with billing enabled
2. Qdrant instance (either managed or self-hosted)
3. Google Cloud Storage bucket for images
4. Pub/Sub topic for image notifications

### Environment Variables

The cloud function requires the following environment variables:

```
GCS_BUCKET_NAME=tagger-images
QDRANT_URL=your-qdrant-host-or-ip
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=image-vectors
```

### Deployment Steps

1. Install the Google Cloud SDK and authenticate:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. Build and deploy using Docker container (recommended for faster cold starts):

```bash
# Build the container
docker build -t gcr.io/YOUR_PROJECT_ID/image-tagger:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/image-tagger:latest

# Deploy the Cloud Function with container
gcloud functions deploy image-tagger \
  --gen2 \
  --region=us-central1 \
  --docker-repository=gcr.io/YOUR_PROJECT_ID/image-tagger \
  --entry-point=process_pubsub_message \
  --trigger-topic=image-upload-notifications \
  --memory=8GB \
  --timeout=540s \
  --min-instances=0 \
  --max-instances=10 \
  --set-env-vars=GCS_BUCKET_NAME=tagger-images,QDRANT_URL=your-qdrant-host,QDRANT_PORT=6333,QDRANT_API_KEY=your-api-key,QDRANT_COLLECTION=image-vectors
```

3. Alternative: Direct deployment without Docker (not recommended for ML workloads):

```bash
gcloud functions deploy image-tagger \
  --gen2 \
  --runtime=python310 \
  --region=us-central1 \
  --source=. \
  --entry-point=process_pubsub_message \
  --trigger-topic=image-upload-notifications \
  --memory=8GB \
  --timeout=540s \
  --min-instances=0 \
  --max-instances=10 \
  --set-env-vars=GCS_BUCKET_NAME=tagger-images,QDRANT_URL=your-qdrant-host,QDRANT_PORT=6333,QDRANT_API_KEY=your-api-key,QDRANT_COLLECTION=image-vectors
```

## Pub/Sub Message Format

The Cloud Function expects messages in the following JSON format:

```json
{
  "filepath": "/path/to/your/image.jpg",
  "timestamp": "2023-01-01T12:00:00Z",
  "image": "base64-encoded-image-data"
}
```

## Development

### Local Testing

There are two ways to test the function locally:

#### Option 1: Using Docker (recommended)

1. Build the Docker image:

```bash
docker build -t image-tagger:local .
```

2. Run the container:

```bash
docker run -p 8080:8080 \
  -e GCS_BUCKET_NAME=tagger-images \
  -e QDRANT_URL=host.docker.internal \
  -e QDRANT_PORT=6333 \
  -e QDRANT_COLLECTION=image-vectors \
  image-tagger:local
```

3. Send a test message:

#### Option 2: Using Functions Framework directly

1. Install dependencies:

```bash
pip install -r requirements-cloud.txt
```

2. Set environment variables:

```bash
export GCS_BUCKET_NAME=tagger-images
export QDRANT_URL=localhost
export QDRANT_PORT=6333
export QDRANT_COLLECTION=image-vectors
```

3. Run the Function Framework:

```bash
functions-framework --target=process_pubsub_message --signature-type=cloudevent
```

4. Send a test message:

```bash
curl localhost:8080 \
  -H "Content-Type: application/json" \
  -H "ce-id: 123451234512345" \
  -H "ce-specversion: 1.0" \
  -H "ce-time: 2020-01-02T12:34:56.789Z" \
  -H "ce-type: google.cloud.pubsub.topic.v1.messagePublished" \
  -H "ce-source: //pubsub.googleapis.com/projects/MY-PROJECT/topics/MY-TOPIC" \
  -d '{
    "message": {
      "data": "BASE64_ENCODED_JSON_MESSAGE"
    },
    "subscription": "projects/MY-PROJECT/subscriptions/MY-SUB"
  }'
```

## Extending with Additional Vector Models

To add a new vector model:

1. Create a new module with your vector generation logic
2. Import it in main.py
3. Call the new vector generation function in the main handler

Example for adding CLIP vectors:

```python
# In vector_generator.py
def generate_clip_vector(image_data, input_data):
    # Implementation here
    return clip_vector_data

# In main.py
from vector_generator import generate_qdrant_sparse_vector, generate_clip_vector, initialize_qdrant_client

# Add to the cloud function
clip_vector = generate_clip_vector(image_data, result)
```