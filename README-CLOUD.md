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

The cloud service requires the following environment variables (see `.env.example`):

```
# Google Cloud Platform configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=your-bucket-name

# Qdrant configuration
QDRANT_URL=your-qdrant-instance.region.gcp.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=image-vectors

# Pub/Sub configuration
PUBSUB_TOPIC=image-upload-notifications
PUBSUB_SUBSCRIPTION=image-tagger-sub

# Cloud Function / Cloud Run configuration  
SERVICE_NAME=image-tagger-run
SERVICE_ACCOUNT=your-service-account@your-project-id.iam.gserviceaccount.com
MEMORY=4Gi
CPU=2
TIMEOUT=540s
```

### Deployment Steps

1. Install the Google Cloud SDK and authenticate:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. Enable required Google Cloud APIs:

```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com pubsub.googleapis.com storage.googleapis.com
```

3. Create Artifact Registry repository:

```bash
gcloud artifacts repositories create image-tagger --repository-format=docker --location=us-central1
```

4. Create Cloud Storage bucket and Pub/Sub topic:

```bash
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME
gcloud pubsub topics create image-upload-notifications
```

5. Create service account (or use an existing one):

```bash
# Optional: Create a new service account
gcloud iam service-accounts create image-tagger-sa

# Grant required permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:image-tagger-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:image-tagger-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/pubsub.subscriber"
```

6. Build and push Docker container:

```bash
# Build the container
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/image-tagger/image-tagger:latest .

# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/image-tagger/image-tagger:latest
```

7. Configure the Qdrant collection for wavelet hash (required):

```bash
# Create or update Qdrant collection with wavelet hash as primary vector
source .env && python scripts/manage_qdrant.py create --overwrite
```

8. Deploy to Cloud Run:

```bash
# Using environment variables from .env file (single-line command)
source /Users/allenday/src/wd14-tagger-standalone/.env && gcloud run deploy $SERVICE_NAME --image=us-central1-docker.pkg.dev/$GCP_PROJECT_ID/image-tagger/image-tagger:latest --region=$GCP_REGION --project=$GCP_PROJECT_ID --memory=$MEMORY --cpu=$CPU --timeout=$TIMEOUT --service-account=$SERVICE_ACCOUNT --set-env-vars=GCS_BUCKET_NAME=$GCS_BUCKET_NAME,QDRANT_URL=$QDRANT_URL,QDRANT_PORT=$QDRANT_PORT,QDRANT_API_KEY=$QDRANT_API_KEY,QDRANT_COLLECTION=$QDRANT_COLLECTION --no-allow-unauthenticated
```

Or with explicit values:

```bash
gcloud run deploy YOUR_SERVICE_NAME --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/image-tagger/image-tagger:latest --region=YOUR_REGION --project=YOUR_PROJECT_ID --memory=4Gi --cpu=2 --timeout=540s --service-account=YOUR_SERVICE_ACCOUNT --set-env-vars=GCS_BUCKET_NAME=YOUR_BUCKET_NAME,QDRANT_URL=YOUR_QDRANT_URL,QDRANT_PORT=6333,QDRANT_API_KEY=YOUR_QDRANT_API_KEY,QDRANT_COLLECTION=image-vectors --no-allow-unauthenticated
```

8. Create a Pub/Sub subscription that pushes to the Cloud Run service:

```bash
gcloud pubsub subscriptions create YOUR_SUBSCRIPTION_NAME \
  --topic=image-upload-notifications \
  --project=YOUR_PROJECT_ID \
  --push-endpoint=https://YOUR_SERVICE_NAME-YOUR_PROJECT_NUMBER.YOUR_REGION.run.app \
  --push-auth-service-account=YOUR_SERVICE_ACCOUNT
```

### Testing the Deployment

You can publish an image directly to Pub/Sub using the included script:

```bash
# Publish directly with the script
python3 publish_to_pubsub.py /Users/allenday/Pictures/vlcsnap-2025-03-28-20h24m00s164.jpg \
  --project=YOUR_PROJECT_ID \
  --topic=image-upload-notifications \
  --max-size=256
```

You can also create a test message file without publishing (useful for debugging):

```bash
# Create a test message with an image (without publishing)
python3 publish_to_pubsub.py /Users/allenday/Pictures/vlcsnap-2025-03-28-20h24m00s164.jpg \
  --save-only test_message.json
```

And then publish using gcloud:

```bash
# Publish the saved message file using gcloud
gcloud pubsub topics publish image-upload-notifications \
  --message="$(cat test_message.json)" \
  --project=YOUR_PROJECT_ID
```

3. Verify successful processing by checking all components:

### Verification Scripts

We've created helper scripts to verify all aspects of processing. First, make sure you have a `.env` file with all the necessary credentials (see `.env.example`).

#### Checking Everything at Once

To check all aspects of processing with a single command:

```bash
# Load environment variables first
source .env

# Run the comprehensive check script
python scripts/check_all.py $EXAMPLE_IMAGE_PATH

# To look further back in the logs (e.g., 24 hours)
python scripts/check_all.py $EXAMPLE_IMAGE_PATH --hours 24
```

This script runs all verification scripts in sequence, checking:
- Cloud Run logs for processing status
- Google Cloud Storage for the stored image
- Qdrant for the vector representation and tags

#### Checking Individual Components

You can also check each component separately:

##### 1. Cloud Run Logs

```bash
# Make sure environment variables are loaded
source .env

# Check logs for evidence of processing
python scripts/check_logs.py $EXAMPLE_IMAGE_PATH

# Look further back in time
python scripts/check_logs.py $EXAMPLE_IMAGE_PATH --hours 24
```

##### 2. Google Cloud Storage

```bash
source .env
python scripts/check_gcs.py $EXAMPLE_IMAGE_PATH
```

##### 3. Qdrant Vector Database

```bash
source .env
python scripts/check_qdrant.py $EXAMPLE_IMAGE_PATH
```

### Manual Verification

If you prefer to check manually:

#### Using gcloud for Logs

```bash
# General format
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=YOUR_SERVICE_NAME" \
  --project=YOUR_PROJECT_ID \
  --limit=10 \
  --order=desc

# For the actual deployed service
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=image-tagger-run" \
  --project=n8n-dev-453405 \
  --limit=20
```

Look for successful log entries like:
```
INFO:main:Successfully processed image and stored vector for /path/to/image.jpg
```

#### Using gsutil for Google Cloud Storage

```bash
# General format
gsutil ls -l gs://YOUR_BUCKET_NAME

# For the actual deployed service
gsutil ls -l gs://camie-tagger
```

Images are stored with a path structure based on the original filename:
```
gs://camie-tagger/vlcsnap-2025-03-28-20h24m00s164/vlcsnap-2025-03-28-20h24m00s164.jpg
```


### Troubleshooting Common Issues

- **Deployment Failures**:
  - Ensure all necessary APIs are enabled
  - Check Docker image builds successfully
  - Verify service account has required permissions

- **Container Startup Failures**:
  - Check logs for missing dependencies
  - For OpenCV issues, ensure system libraries are installed
  - Verify environment variables are correctly set

- **Message Processing Failures**:
  - Check Pub/Sub subscription configuration
  - Verify authentication between Pub/Sub and Cloud Run
  - Examine Cloud Run logs for detailed error messages

## Pub/Sub Message Format

The Cloud Function requires a specific format for Pub/Sub messages. The publish_to_pubsub.py script handles this properly, but if you need to create test messages manually, here's the structure:

1. Inner message (the actual data):
```json
{
  "filepath": "/Users/allenday/Pictures/vlcsnap-2025-03-28-20h24m00s164.jpg",
  "timestamp": "2023-01-01T12:00:00Z",
  "image": "base64-encoded-image-data"
}
```

2. This inner message must be converted to JSON and base64 encoded

3. Then wrapped in a Cloud Events compatible structure:
```json
{
  "message": {
    "data": "base64-encoded-json-from-step-2"
  }
}
```

The double-encoding and wrapping is handled automatically by the publish_to_pubsub.py script.

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