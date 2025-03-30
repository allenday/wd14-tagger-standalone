#!/usr/bin/env python3
"""
Publish an image to Google Cloud Pub/Sub for processing by the image-tagger Cloud Function.
This script properly formats the message with the double-encoding required by Cloud Functions
that process Pub/Sub messages via Cloud Events.
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path
from PIL import Image
from io import BytesIO
from google.cloud import pubsub_v1

def resize_image(image_path, max_size=512):
    """Resize an image while maintaining aspect ratio."""
    img = Image.open(image_path)
    # Resize image while maintaining aspect ratio
    img.thumbnail((max_size, max_size))
    # Save to BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    image_data = buffer.getvalue()
    print(f"Resized image to {img.width}x{img.height} ({len(image_data)/1024:.1f} KB)")
    return image_data

def publish_message(project_id, topic_id, image_path, max_size=512, filepath=None):
    """Publish a message to Pub/Sub with the image data in Cloud Function format.
    
    Args:
        project_id: Google Cloud project ID
        topic_id: Pub/Sub topic ID
        image_path: Local path to the image file to publish
        max_size: Maximum dimension for image resizing
        filepath: Optional custom filepath to include in the message payload. 
                 If not provided, the actual image_path will be used.
    """
    # Check if file exists
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        print(f"Error: File {image_path} does not exist or is not a file")
        return False
    
    # Resize image - smaller than default to ensure message isn't too large
    image_data = resize_image(image_path, max_size)
    
    # Step 1: Create inner message with image data
    inner_message = {
        "filepath": filepath if filepath else str(path),
        "timestamp": "2023-01-01T12:00:00Z",
        "image": base64.b64encode(image_data).decode("utf-8")
    }
    
    # Step 2: Convert inner message to JSON and encode (first encoding)
    inner_json = json.dumps(inner_message)
    encoded_data = base64.b64encode(inner_json.encode("utf-8")).decode("utf-8")
    
    # Step 3: Create Cloud Events compatible message (wrapping the encoded data)
    pubsub_message = {
        "message": {
            "data": encoded_data
        }
    }
    
    # Step 4: Convert to final JSON for sending
    message_data = json.dumps(pubsub_message).encode("utf-8")
    
    print(f"Message size: {len(message_data)/1024:.1f} KB")
    
    # Publish to Pub/Sub
    try:
        print(f"Connecting to Pub/Sub topic {project_id}/{topic_id}...")
        
        # Create a publisher client with explicit timeout settings
        client_options = {
            "api_endpoint": "pubsub.googleapis.com:443"
        }
        publisher = pubsub_v1.PublisherClient(client_options=client_options)
        topic_path = publisher.topic_path(project_id, topic_id)
        
        # Check if topic exists first
        try:
            publisher.get_topic(request={"topic": topic_path})
            print(f"Topic {topic_path} exists")
        except Exception as e:
            print(f"Warning: Could not verify topic exists: {e}")
        
        print(f"Publishing message to {topic_path}...")
        
        # Set up callback for publish
        def callback(future):
            try:
                message_id = future.result(timeout=60)
                print(f"Published message ID: {message_id}")
            except Exception as e:
                print(f"Publishing failed: {e}")
        
        # Publish asynchronously
        future = publisher.publish(topic_path, message_data)
        future.add_done_callback(callback)
        
        # Wait for the publish to complete
        print("Waiting for publish to complete...")
        message_id = future.result(timeout=60)
        print(f"Published message ID: {message_id}")
        print(f"Message sent to {topic_path}")
        return True
        
    except Exception as e:
        import traceback
        print(f"Error publishing message: {e}")
        print(traceback.format_exc())
        return False

def create_test_message(image_path, max_size=512, output_file=None, filepath=None):
    """Create a test message file without publishing.
    
    Args:
        image_path: Local path to the image file to publish
        max_size: Maximum dimension for image resizing
        output_file: Optional file to save the test message to
        filepath: Optional custom filepath to include in the message payload.
                 If not provided, the actual image_path will be used.
    """
    # Check if file exists
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        print(f"Error: File {image_path} does not exist or is not a file")
        return False
    
    # Resize image
    image_data = resize_image(image_path, max_size)
    
    # Step 1: Create inner message with image data
    inner_message = {
        "filepath": filepath if filepath else str(path),
        "timestamp": "2023-01-01T12:00:00Z",
        "image": base64.b64encode(image_data).decode("utf-8")
    }
    
    # Step 2: Convert inner message to JSON and encode (first encoding)
    inner_json = json.dumps(inner_message)
    encoded_data = base64.b64encode(inner_json.encode("utf-8")).decode("utf-8")
    
    # Step 3: Create Cloud Events compatible message (wrapping the encoded data)
    pubsub_message = {
        "message": {
            "data": encoded_data
        }
    }
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(pubsub_message, f)
        print(f"Test message saved to {output_file}")
    
    return pubsub_message

def check_credentials():
    """Check if Google Cloud credentials are properly set up."""
    try:
        import google.auth
        credentials, project = google.auth.default()
        print(f"Found credentials for project: {project}")
        return True
    except Exception as e:
        print(f"Error with Google Cloud credentials: {e}")
        print("\nTry setting up credentials with one of these methods:")
        print("1. Run 'gcloud auth application-default login'")
        print("2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to point to a service account key file")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Publish an image to Pub/Sub in Cloud Function format",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--project", help="Google Cloud project ID")
    parser.add_argument("--topic", help="Pub/Sub topic ID")
    parser.add_argument("--filepath", help="Custom filepath to include in the message payload (different from the actual file path)")
    parser.add_argument("--max-size", type=int, default=256, 
                        help="Maximum dimension for resizing (default: 256)")
    parser.add_argument("--check-auth", action="store_true", 
                        help="Check Google Cloud authentication only")
    parser.add_argument("--save-only", metavar="FILE", 
                        help="Create a test message JSON file without publishing")
    
    args = parser.parse_args()
    
    # Check credentials first
    if args.check_auth:
        check_credentials()
        return
    
    # Check if file exists
    path = Path(args.image_path)
    if not path.exists() or not path.is_file():
        print(f"Error: File {args.image_path} does not exist or is not a file")
        sys.exit(1)
    
    # Save-only mode
    if args.save_only:
        create_test_message(args.image_path, args.max_size, args.save_only, args.filepath)
        return
    
    # Validate required args for publishing
    if not args.project or not args.topic:
        print("Error: --project and --topic are required for publishing to Pub/Sub")
        print("Use --save-only to create a test message without publishing")
        sys.exit(1)
    
    if not check_credentials():
        print("Authentication check failed. Fix credentials before continuing.")
        return
    
    # Publish message
    success = publish_message(
        args.project, 
        args.topic, 
        args.image_path, 
        args.max_size,
        args.filepath
    )
    
    if success:
        print(f"Successfully published image {args.image_path} to Pub/Sub")
    else:
        print(f"Failed to publish image to Pub/Sub")
        
if __name__ == "__main__":
    main()