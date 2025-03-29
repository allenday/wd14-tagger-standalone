#!/usr/bin/env python3
"""
Test the full image processing workflow by:
1. Publishing an image to Pub/Sub
2. Checking for the image in Google Cloud Storage
3. Checking for the vector in Qdrant
"""
import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import script modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.check_gcs import check_gcs
from scripts.check_qdrant import check_vector
from scripts.publish_to_pubsub import publish_message, create_test_message, check_credentials

def load_env():
    """Load environment variables from .env file."""
    # First try .env in the current directory
    if os.path.exists(".env"):
        load_dotenv(".env")
    # Then try .env in the parent directory (if running from scripts/)
    elif os.path.exists("../.env"):
        load_dotenv("../.env")
    
    # Check required variables
    required_vars = ["GCP_PROJECT_ID", "PUBSUB_TOPIC", "GCS_BUCKET_NAME", "QDRANT_COLLECTION"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("Make sure your .env file has these variables set.")
        sys.exit(1)
    
    return {
        "project_id": os.getenv("GCP_PROJECT_ID"),
        "topic_id": os.getenv("PUBSUB_TOPIC"),
        "bucket_name": os.getenv("GCS_BUCKET_NAME"),
        "collection_name": os.getenv("QDRANT_COLLECTION")
    }

def test_workflow(image_path, max_retries=5, delay=10):
    """Test the full image processing workflow."""
    # Load environment variables
    env = load_env()
    
    # Check credentials
    if not check_credentials():
        print("Authentication check failed. Fix credentials before continuing.")
        return False
    
    # Check if image exists
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        print(f"Error: Image {image_path} does not exist")
        return False
    
    # Step 1: Publish image to Pub/Sub
    print(f"\n-----\nStep 1: Publishing image to Pub/Sub\n-----")
    success = publish_message(
        project_id=env["project_id"],
        topic_id=env["topic_id"],
        image_path=str(path),
        max_size=512  # Resize to 512px max dimension
    )
    
    if not success:
        print("Failed to publish image to Pub/Sub")
        return False
    
    print(f"Published image to Pub/Sub topic {env['topic_id']}")
    print("Waiting for Cloud Run to process the image...")
    
    # Step 2: Wait and check GCS
    print(f"\n-----\nStep 2: Checking for image in Google Cloud Storage\n-----")
    found_in_gcs = False
    
    for attempt in range(max_retries):
        # Wait before checking
        print(f"Waiting {delay} seconds before checking GCS (attempt {attempt+1}/{max_retries})...")
        time.sleep(delay)
        
        # Check GCS
        found_in_gcs = check_gcs(str(path), env["bucket_name"])
        
        if found_in_gcs:
            print("Image found in GCS!")
            break
        
        print(f"Image not found in GCS yet. Retrying ({attempt+1}/{max_retries})...")
    
    if not found_in_gcs:
        print("Image was not found in GCS after all retries")
        return False
    
    # Step 3: Check Qdrant
    print(f"\n-----\nStep 3: Checking for vector in Qdrant\n-----")
    found_in_qdrant = False
    
    for attempt in range(max_retries):
        # Wait before checking if we didn't just check GCS
        if attempt > 0 or not found_in_gcs:
            print(f"Waiting {delay} seconds before checking Qdrant (attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
        
        # Check Qdrant
        found_in_qdrant = check_vector(str(path), {"collection": env["collection_name"]})
        
        if found_in_qdrant:
            print("Vector found in Qdrant!")
            break
        
        print(f"Vector not found in Qdrant yet. Retrying ({attempt+1}/{max_retries})...")
    
    if not found_in_qdrant:
        print("Vector was not found in Qdrant after all retries")
        return False
    
    # Success!
    print("\n-----\nWorkflow test completed successfully!\n-----")
    print("✅ Image published to Pub/Sub")
    print("✅ Image stored in Google Cloud Storage")
    print("✅ Vector stored in Qdrant")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test the full image processing workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--retries", type=int, default=5, help="Number of times to check for results")
    parser.add_argument("--delay", type=int, default=10, help="Seconds to wait between checks")
    
    args = parser.parse_args()
    
    success = test_workflow(args.image_path, args.retries, args.delay)
    
    if success:
        print("\nAll steps completed successfully!")
    else:
        print("\nWorkflow test failed at one or more steps.")
        sys.exit(1)

if __name__ == "__main__":
    main()