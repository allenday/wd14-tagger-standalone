#!/usr/bin/env python3
"""
Script to check if an image was successfully uploaded to Google Cloud Storage.
"""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    from google.cloud import storage
except ImportError:
    print("Error: google-cloud-storage is not installed. Please run: pip install google-cloud-storage")
    sys.exit(1)

def load_env():
    """Load environment variables from .env file."""
    # First try .env in the current directory
    if os.path.exists(".env"):
        load_dotenv(".env")
    # Then try .env in the parent directory (if running from scripts/)
    elif os.path.exists("../.env"):
        load_dotenv("../.env")
    
    # Check required variables
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        print("Error: Missing GCS_BUCKET_NAME environment variable")
        print("Make sure the .env file exists and contains GCS_BUCKET_NAME=your-bucket-name")
        sys.exit(1)
    
    return bucket_name

def check_gcs(filepath, bucket_name, custom_filepath=None):
    """Check if an image exists in GCS.
    
    Args:
        filepath: The local filepath that was processed
        bucket_name: The GCS bucket name
        custom_filepath: Optional custom filepath that was used in the message
    """
    # Use custom filepath if provided, otherwise use the original filepath
    path_to_check = Path(custom_filepath if custom_filepath else filepath)
    filename = path_to_check.name
    parent_path = path_to_check.parent
    
    # Expected GCS path based on the Cloud Function logic
    if str(parent_path) == '/' or str(parent_path) == '.':
        # If no parent directory, use stem as directory
        expected_blob_name = f"{path_to_check.stem}/{filename}"
    else:
        # Use the full parent path
        parent_str = str(parent_path)
        if parent_str.startswith('/'):
            parent_str = parent_str[1:]
        expected_blob_name = f"{parent_str}/{filename}"
    
    print(f"Checking for GCS object: gs://{bucket_name}/{expected_blob_name}")
    
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(expected_blob_name)
        
        # Check if blob exists
        if blob.exists():
            print("✅ Image found in Google Cloud Storage!")
            print(f"Public URL: {blob.public_url}")
            
            # Get metadata
            blob.reload()  # Refresh to get metadata
            metadata = {k: v for k, v in blob.metadata.items()} if blob.metadata else {}
            
            print("\nMetadata:")
            if metadata:
                for key, value in metadata.items():
                    print(f"{key}: {value}")
            else:
                print("No custom metadata found.")
            
            print(f"\nSize: {blob.size/1024:.2f} KB")
            print(f"Content type: {blob.content_type}")
            print(f"Created: {blob.time_created}")
            print(f"Updated: {blob.updated}")
            
            return True
        else:
            print("❌ Image not found in Google Cloud Storage.")
            print("This could mean:")
            print("1. The image hasn't been processed yet.")
            print("2. The processing failed.")
            print("3. The Cloud Function used a different path structure.")
            
            # Check for any blobs with similar names
            print("\nSearching for alternative paths...")
            blobs = list(bucket.list_blobs(prefix=stem))
            if blobs:
                print(f"Found {len(blobs)} possible matches:")
                for b in blobs:
                    print(f"- gs://{bucket_name}/{b.name}")
            else:
                print("No alternative paths found.")
            
            return False
    
    except Exception as e:
        print(f"Error checking GCS: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check if an image was uploaded to Google Cloud Storage")
    parser.add_argument("filepath", help="Path to the image file to check")
    
    args = parser.parse_args()
    
    # Check if file exists
    path = Path(args.filepath)
    if not path.exists() or not path.is_file():
        print(f"Error: File {args.filepath} does not exist or is not a file")
        sys.exit(1)
    
    # Load environment variables
    bucket_name = load_env()
    
    # Check GCS
    check_gcs(str(path), bucket_name)

if __name__ == "__main__":
    main()