import base64
import json
import os
import tempfile
from typing import Dict, Any
from pathlib import Path
import uuid
import logging
from io import BytesIO

from google.cloud import storage
import functions_framework
from PIL import Image

from tagger.interrogator import AbsInterrogator
from tagger.interrogators import interrogators
from vector_generator import generate_qdrant_sparse_vector, initialize_qdrant_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to cache model and other resources
interrogator_instance = None
qdrant_client = None
bucket_client = None

def initialize_model(model_name="camie-tagger"):
    """Initialize and cache the model."""
    global interrogator_instance
    
    if interrogator_instance is None:
        logger.info(f"Initializing {model_name} model...")
        interrogator_instance = interrogators[model_name]
        # Always use CPU for cloud functions
        interrogator_instance.override_execution_provider(['CPUExecutionProvider'])
        logger.info(f"Model {model_name} initialized successfully")
    
    return interrogator_instance

def initialize_storage():
    """Initialize and cache the GCS client."""
    global bucket_client
    
    if bucket_client is None:
        logger.info("Initializing Storage client...")
        bucket_client = storage.Client()
        logger.info("Storage client initialized")
    
    return bucket_client

def process_image(image_data, threshold=0.2):
    """Process image and get tags with confidence scores."""
    model = initialize_model()
    
    # Create PIL Image from bytes
    image = Image.open(BytesIO(image_data))
    
    # Run inference
    result = model.interrogate(image)
    
    # Extract tags and apply threshold
    tags = AbsInterrogator.postprocess_tags(
        result[1],
        threshold=threshold,
        escape_tag=False,
        replace_underscore=False,
        exclude_tags=set()
    )
    
    return tags

def save_to_gcs(bucket_name, destination_blob_name, image_data):
    """Upload image to Google Cloud Storage."""
    storage_client = initialize_storage()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_string(image_data, content_type="image/jpeg")
    
    logger.info(f"Image uploaded to gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"

@functions_framework.cloud_event
def process_pubsub_message(cloud_event):
    """Cloud Function entry point - processes Pub/Sub messages."""
    # Get Pub/Sub message
    pubsub_message = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    
    try:
        # Parse the message content
        message_data = json.loads(pubsub_message)
        
        # Extract fields from the message
        filepath = message_data.get("filepath")
        timestamp = message_data.get("timestamp")
        image_base64 = message_data.get("image")
        
        if not filepath or not image_base64:
            logger.error("Missing required fields in message: filepath and/or image")
            return
        
        # Decode the image
        image_data = base64.b64decode(image_base64)
        
        # Create storage path from filepath
        # This can be customized based on your needs
        filename = Path(filepath).name
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "tagger-images")
        destination_blob_name = f"{Path(filepath).stem}/{filename}"
        
        # Save image to GCS
        gcs_uri = save_to_gcs(bucket_name, destination_blob_name, image_data)
        
        # Process the image and get tags
        tags = process_image(image_data)
        
        # Create result object
        result = {
            "file": filepath,
            "gcs_uri": gcs_uri,
            "timestamp": timestamp,
            "tags": tags
        }
        
        # Generate vector representation
        vector_data = generate_qdrant_sparse_vector(result)
        
        # Store in Qdrant
        qdrant = initialize_qdrant_client()
        collection_name = os.environ.get("QDRANT_COLLECTION", "image-vectors")
        
        # Use filepath as ID (hash it for consistency)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))
        
        # Insert into Qdrant
        qdrant.upsert(
            collection_name=collection_name,
            points=[{
                "id": point_id,
                "vector": vector_data["vector"],
                "payload": vector_data["payload"]
            }]
        )
        
        logger.info(f"Successfully processed image and stored vector for {filepath}")
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise