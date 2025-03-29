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
    # Debug log for diagnosing message format issues
    logger.info(f"Received cloud event with data keys: {cloud_event.data.keys() if hasattr(cloud_event, 'data') and cloud_event.data else 'No data'}")
    logger.info(f"Cloud event data type: {type(cloud_event.data) if hasattr(cloud_event, 'data') else 'No data attribute'}")
    
    # Get Pub/Sub message with error handling
    try:
        # First, safely access the nested message data
        if not hasattr(cloud_event, 'data') or not cloud_event.data:
            logger.error("No data in cloud event")
            return
            
        if "message" not in cloud_event.data:
            logger.error(f"No 'message' in cloud event data. Keys: {cloud_event.data.keys()}")
            return
            
        message = cloud_event.data.get("message", {})
        if not message or "data" not in message:
            logger.error(f"No 'data' in message. Message: {message}")
            return
        
        # Base64 decode the message data
        encoded_data = message["data"]
        logger.info(f"Encoded data has length: {len(encoded_data)}")
        pubsub_message = base64.b64decode(encoded_data).decode('utf-8')
        logger.info(f"Decoded message starts with: {pubsub_message[:100]}...")
    
        # Parse the message content
        message_data = json.loads(pubsub_message)
        logger.info(f"Successfully parsed JSON with keys: {message_data.keys()}")
        
        # Extract fields from the message
        filepath = message_data.get("filepath")
        timestamp = message_data.get("timestamp")
        image_base64 = message_data.get("image")
        
        logger.info(f"Processing image from: {filepath}")
        
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
        
        # Insert into Qdrant with proper format for sparse vectors
        # The Python client doesn't support sparse_vectors field yet, but the REST API does
        # So we'll drop the sparse vectors for now and just use the dense vector
        try:
            # First try without sparse vectors
            logger.info(f"Inserting point {point_id} into Qdrant collection {collection_name}")
            qdrant.upsert(
                collection_name=collection_name,
                points=[{
                    "id": point_id,
                    "vector": vector_data["vector"],  # This is the dummy dense vector
                    "payload": vector_data["payload"]
                }]
            )
            logger.info("Vector insertion successful")
            
            # Store sparse vector information in the payload for now
            if "sparse_vectors" in vector_data and vector_data["sparse_vectors"]:
                logger.info("Sparse vector information included in payload")
                # We'll add this to the payload instead since the client doesn't support sparse_vectors
                qdrant.set_payload(
                    collection_name=collection_name,
                    payload={"_sparse_vectors": vector_data["sparse_vectors"]},
                    points=[point_id]
                )
                logger.info("Updated payload with sparse vector information")
        except Exception as e:
            logger.error(f"Error inserting into Qdrant: {str(e)}")
            # Try again with minimal format
            try:
                logger.info("Retrying with minimal format")
                qdrant.upsert(
                    collection_name=collection_name,
                    points=[{
                        "id": point_id,
                        "vector": vector_data["vector"],
                        "payload": {
                            "file": result["file"],
                            "gcs_uri": result["gcs_uri"],
                            "timestamp": result["timestamp"],
                            "tags": result["tags"]
                        }
                    }]
                )
                logger.info("Minimal vector insertion successful")
            except Exception as e2:
                logger.error(f"Error with minimal format: {str(e2)}")
                raise
        
        logger.info(f"Successfully processed image and stored vector for {filepath}")
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Message content: {pubsub_message[:200]}...")
        return  # Don't raise, just log and return
    except KeyError as e:
        logger.error(f"Missing key in message: {str(e)}")
        return  # Don't raise, just log and return
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return  # Don't raise, just log and return