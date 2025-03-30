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

# Pre-load the model during module initialization to avoid cold-start latency
try:
    # Only pre-load if we're not in a build context (Docker build)
    # This avoids errors during container build
    if os.environ.get("IN_DOCKER_BUILD") != "1":
        logger.info("Pre-loading model for faster cold start...")
        
        # Check the HuggingFace cache directory to see if files are available
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        logger.info(f"HuggingFace cache directory: {hf_cache_dir}")
        if os.path.exists(hf_cache_dir):
            cache_contents = os.listdir(hf_cache_dir)
            logger.info(f"HuggingFace cache contents: {cache_contents}")
            
            model_dir = os.path.join(hf_cache_dir, "models--Camais03--camie-tagger")
            if os.path.exists(model_dir):
                logger.info(f"Found cached model directory: {model_dir}")
                if os.path.exists(os.path.join(model_dir, "blobs")):
                    logger.info(f"Blobs directory exists with contents: {os.listdir(os.path.join(model_dir, 'blobs'))}")
                if os.path.exists(os.path.join(model_dir, "snapshots", "latest")):
                    logger.info(f"Latest snapshot directory exists with contents: {os.listdir(os.path.join(model_dir, 'snapshots', 'latest'))}")
            else:
                logger.warning(f"Model directory not found in cache: {model_dir}")
        else:
            logger.warning(f"HuggingFace cache directory does not exist: {hf_cache_dir}")
        
        # Initialize the model
        interrogator_instance = interrogators["camie-tagger"]
        interrogator_instance.override_execution_provider(['CPUExecutionProvider'])
        
        # Force pre-load the model by calling load()
        interrogator_instance.load()
        
        logger.info("Model pre-loaded successfully")
    else:
        logger.info("Skipping model pre-load during Docker build")
except Exception as e:
    logger.error(f"Error pre-loading model: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    # Continue anyway, we'll try to load it again when needed

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
        
        # Check if we're dealing with a nested structure (due to double-encoding)
        if "message" in message_data and "data" not in message_data:
            logger.info("Detected nested message format, fixing...")
            # If we have a nested message structure, we need to decode one more level
            try:
                inner_data = message_data.get("message", {}).get("data")
                if inner_data:
                    inner_json = base64.b64decode(inner_data).decode('utf-8')
                    message_data = json.loads(inner_json)
                    logger.info(f"Successfully parsed inner JSON with keys: {message_data.keys()}")
            except Exception as e:
                logger.error(f"Error parsing nested message: {str(e)}")
                return
        
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
        
        # Handle different formats of filepath
        if filepath.startswith("https://storage.cloud.google.com/"):
            # Extract bucket and object path from HTTPS URL
            # Format: https://storage.cloud.google.com/bucket-name/path/to/object
            https_path = filepath.replace("https://storage.cloud.google.com/", "")
            parts = https_path.split("/", 1)
            if len(parts) == 2:
                bucket_name = parts[0]
                destination_blob_name = parts[1]
                logger.info(f"Extracted from HTTPS URL - Bucket: {bucket_name}, Path: {destination_blob_name}")
            else:
                logger.error(f"Invalid HTTPS GCS URL format: {filepath}")
                # Use default values as fallback
                bucket_name = os.environ.get("GCS_BUCKET_NAME", "tagger-images")
                destination_blob_name = f"unknown/{Path(filepath).name}"
        elif filepath.startswith("gs://"):
            # Extract bucket and object path from gs:// URL
            gs_path = filepath[5:]  # Remove "gs://"
            parts = gs_path.split("/", 1)
            if len(parts) == 2:
                bucket_name = parts[0]
                destination_blob_name = parts[1]
                logger.info(f"Extracted from GS URL - Bucket: {bucket_name}, Path: {destination_blob_name}")
            else:
                logger.error(f"Invalid GS URL format: {filepath}")
                # Use default values as fallback
                bucket_name = os.environ.get("GCS_BUCKET_NAME", "tagger-images")
                destination_blob_name = f"unknown/{Path(filepath).name}"
        else:
            # Regular filepath, preserve directory structure
            path = Path(filepath)
            filename = path.name
            parent_path = path.parent
            
            # Get bucket name from environment
            bucket_name = os.environ.get("GCS_BUCKET_NAME", "tagger-images")
            
            # If parent path is just / or empty, use the stem as the directory
            # Otherwise use the full parent path
            if str(parent_path) == '/' or str(parent_path) == '.':
                destination_blob_name = f"{path.stem}/{filename}"
            else:
                # Remove leading / if present to avoid empty segment at the beginning
                parent_str = str(parent_path)
                if parent_str.startswith('/'):
                    parent_str = parent_str[1:]
                    
                destination_blob_name = f"{parent_str}/{filename}"
        
        logger.info(f"Storage path: {bucket_name}/{destination_blob_name}")
        
        # Save image to GCS
        gcs_uri = save_to_gcs(bucket_name, destination_blob_name, image_data)
        
        # Process the image and get tags
        tags = process_image(image_data)
        
        # Convert GCS URI to HTTPS URL for the result object
        https_uri = gcs_uri
        if gcs_uri.startswith("gs://"):
            # Format: gs://bucket-name/path/to/object
            gs_path = gcs_uri[5:]  # Remove "gs://"
            parts = gs_path.split("/", 1)
            if len(parts) == 2:
                bucket_name, object_path = parts
                https_uri = f"https://storage.cloud.google.com/{bucket_name}/{object_path}"
                logger.info(f"Converted GCS URI to HTTPS URL for storage: {https_uri}")
        
        # Create result object with both the original filepath and HTTPS URL
        result = {
            "file": filepath,      # Keep the original filepath
            "gcs_uri": gcs_uri,    # Keep the original GCS URI (gs:// format)
            "https_uri": https_uri, # Add the HTTPS URL for direct browsing
            "timestamp": timestamp,
            "tags": tags
        }
        
        # Generate vector representation including wavelet hash
        vector_data = generate_qdrant_sparse_vector(result, image_data=image_data, include_whash=True)
        
        # Store in Qdrant
        qdrant = initialize_qdrant_client()
        collection_name = os.environ.get("QDRANT_COLLECTION", "image-vectors")
        
        # Use filepath as ID (hash it for consistency)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))
        
        # Insert into Qdrant with proper format for sparse vectors and wavelet hash
        try:
            # Check if we have the combined format with tag vector and whash vector
            if "vectors" in vector_data and "whash_vector" in vector_data["vectors"]:
                logger.info(f"Inserting point {point_id} with tag vector and wavelet hash")
                # Use the new format with named vectors
                qdrant.upsert(
                    collection_name=collection_name,
                    points=[{
                        "id": point_id,
                        "vectors": {
                            "tag_vector": vector_data["vectors"]["tag_vector"],
                            "whash_vector": vector_data["vectors"]["whash_vector"]
                        },
                        "payload": vector_data["payload"]
                    }]
                )
                logger.info("Vector insertion with tag vector and wavelet hash successful")
                
                # Store sparse vector information in the payload
                if "sparse_vectors" in vector_data and vector_data["sparse_vectors"]:
                    logger.info("Adding sparse vector information to payload")
                    qdrant.set_payload(
                        collection_name=collection_name,
                        payload={"_sparse_vectors": vector_data["sparse_vectors"]},
                        points=[point_id]
                    )
                    logger.info("Updated payload with sparse vector information")
            else:
                # Fall back to the old format if wavelet hash is not available
                logger.info(f"Inserting point {point_id} with tag vector only")
                qdrant.upsert(
                    collection_name=collection_name,
                    points=[{
                        "id": point_id,
                        "vector": vector_data["vector"],  # This is the dummy dense vector
                        "payload": vector_data["payload"]
                    }]
                )
                logger.info("Vector insertion successful")
                
                # Store sparse vector information in the payload
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
                        "vectors": {
                            "tag_vector": [0.0] * 100,  # Use dummy tag vector
                            "whash_vector": [0.0] * 256  # Use dummy wavelet hash vector
                        },
                        "payload": {
                            "file": result["file"],
                            "gcs_uri": result["gcs_uri"],
                            "https_uri": result["https_uri"],
                            "timestamp": result["timestamp"],
                            "tags": result["tags"]
                        }
                    }]
                )
                logger.info("Minimal vector insertion successful")
            except Exception as e2:
                logger.error(f"Error with minimal format: {str(e2)}")
                
                # Try one more time with the simplest possible format
                try:
                    logger.info("Retrying with simplest format")
                    qdrant.upsert(
                        collection_name=collection_name,
                        points=[{
                            "id": point_id,
                            "vector": [0.0] * 100,  # Use dummy vector
                            "payload": {
                                "file": result["file"],
                                "gcs_uri": result["gcs_uri"],
                                "https_uri": result["https_uri"],
                                "timestamp": result["timestamp"],
                                "tags": result["tags"]
                            }
                        }]
                    )
                    logger.info("Simplest vector insertion successful")
                except Exception as e3:
                    logger.error(f"Error with simplest format: {str(e3)}")
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