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
from tqdm import tqdm

def resize_image(image_path, max_size=512, verbose=False):
    """Resize an image while maintaining aspect ratio."""
    img = Image.open(image_path)
    # Resize image while maintaining aspect ratio
    img.thumbnail((max_size, max_size))
    # Save to BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    image_data = buffer.getvalue()
    if verbose:
        print(f"Resized image to {img.width}x{img.height} ({len(image_data)/1024:.1f} KB)")
    return image_data

def publish_message(project_id, topic_id, image_path, max_size=512, filepath=None, time_offset=None, verbose=False):
    """Publish a message to Pub/Sub with the image data in Cloud Function format.
    
    Args:
        project_id: Google Cloud project ID
        topic_id: Pub/Sub topic ID
        image_path: Local path to the image file to publish
        max_size: Maximum dimension for image resizing
        filepath: Optional custom filepath to include in the message payload. 
                 If not provided, the actual image_path will be used.
        time_offset: Optional time offset in seconds to include in the message payload.
        verbose: Whether to print detailed progress messages
    """
    # Check if file exists
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        if verbose:
            print(f"Error: File {image_path} does not exist or is not a file")
        return False
    
    # Resize image - smaller than default to ensure message isn't too large
    image_data = resize_image(image_path, max_size, verbose)
    
    # Step 1: Create inner message with image data
    path_str = filepath if filepath else str(path)
    
    # Convert gs:// URLs to https:// URLs if needed
    if path_str.startswith("gs://"):
        # Extract bucket and object path from gs:// URL
        gs_path = path_str[5:]  # Remove "gs://"
        parts = gs_path.split("/", 1)
        if len(parts) == 2:
            bucket_name, object_path = parts
            # Create the HTTPS URL
            path_str = f"https://storage.googleapis.com/{bucket_name}/{object_path}"
            if verbose:
                print(f"Converted GCS path to HTTPS URL: {path_str}")
    
    inner_message = {
        "filepath": path_str,
        "timestamp": "2023-01-01T12:00:00Z",
        "time_offset": time_offset,
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
    
    if verbose:
        print(f"Message size: {len(message_data)/1024:.1f} KB")
    
    # Publish to Pub/Sub
    try:
        if verbose:
            print(f"Connecting to Pub/Sub topic {project_id}/{topic_id}...")
        
        # Create a publisher client with explicit timeout settings
        client_options = {
            "api_endpoint": "pubsub.googleapis.com:443"
        }
        publisher = pubsub_v1.PublisherClient(client_options=client_options)
        topic_path = publisher.topic_path(project_id, topic_id)
        
        # Check if topic exists first (only on first message)
        try:
            publisher.get_topic(request={"topic": topic_path})
            if verbose:
                print(f"Topic {topic_path} exists")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not verify topic exists: {e}")
        
        if verbose:
            print(f"Publishing message to {topic_path}...")
        
        # Publish message
        future = publisher.publish(topic_path, message_data)
        
        # Wait for the publish to complete
        message_id = future.result(timeout=60)
        
        if verbose:
            print(f"Published message ID: {message_id}")
            print(f"Message sent to {topic_path}")
            
        return True
        
    except Exception as e:
        import traceback
        if verbose:
            print(f"Error publishing message: {e}")
            print(traceback.format_exc())
        return False

def create_test_message(image_path, max_size=512, output_file=None, filepath=None, time_offset=None, verbose=False):
    """Create a test message file without publishing.
    
    Args:
        image_path: Local path to the image file to publish
        max_size: Maximum dimension for image resizing
        output_file: Optional file to save the test message to
        filepath: Optional custom filepath to include in the message payload.
                 If not provided, the actual image_path will be used.
        time_offset: Optional time offset in seconds to include in the message payload.
        verbose: Whether to print detailed progress messages
    """
    # Check if file exists
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        if verbose:
            print(f"Error: File {image_path} does not exist or is not a file")
        return False
    
    # Resize image
    image_data = resize_image(image_path, max_size, verbose)
    
    # Step 1: Create inner message with image data
    path_str = filepath if filepath else str(path)
    
    # Convert gs:// URLs to https:// URLs if needed
    if path_str.startswith("gs://"):
        # Extract bucket and object path from gs:// URL
        gs_path = path_str[5:]  # Remove "gs://"
        parts = gs_path.split("/", 1)
        if len(parts) == 2:
            bucket_name, object_path = parts
            # Create the HTTPS URL
            path_str = f"https://storage.googleapis.com/{bucket_name}/{object_path}"
            print(f"Converted GCS path to HTTPS URL: {path_str}")
    
    inner_message = {
        "filepath": path_str,
        "timestamp": "2023-01-01T12:00:00Z",
        "time_offset": time_offset,
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

def find_image_files(directory_path):
    """Find all image files in a directory recursively.
    
    Args:
        directory_path: Path to the directory to search
        
    Returns:
        List of file paths, sorted alphabetically
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = []
    
    # Convert to Path object if it's a string
    path = Path(directory_path)
    
    if path.is_file():
        # If it's a single file, just check if it's an image
        if path.suffix.lower() in image_extensions:
            return [str(path)]
        else:
            return []
            
    print(f"Scanning directory {path} for images...")
    
    # Otherwise, walk the directory
    for root, dirs, files in os.walk(path):
        # Sort directories for consistent traversal
        dirs.sort()
        
        # Sort files in current directory
        sorted_files = sorted(files)
        
        for file in sorted_files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
                
    # Sort the final list for consistent ordering
    image_files.sort()
    
    return image_files

def extract_time_offset(filename, pattern):
    """Extract time offset from a filename based on a pattern.
    
    Args:
        filename: The filename to extract from
        pattern: Pattern string (e.g., 'frame_%05d.jpg')
    
    Returns:
        Time offset as float if detected, None otherwise
    """
    import re
    
    if not pattern:
        return None
        
    # Convert printf-style format to regex pattern
    # e.g., 'frame_%05d.jpg' becomes 'frame_(\d{5})\.jpg'
    regex_pattern = pattern.replace('.', '\.')  # Escape dots
    
    # Replace %d, %03d, %05d, etc. with appropriate regex groups
    regex_pattern = re.sub(r'%(\d*)d', lambda m: f'(\\d{{{m.group(1) or 1},}})', regex_pattern)
    
    # Try to match the pattern against the filename
    match = re.match(regex_pattern, filename)
    if match and match.group(1):
        return float(match.group(1))
    
    return None

def process_path(path_str, base_dir=None, custom_dir=None, time_offset_pattern=None):
    """Process a file path to create the appropriate filepath for publishing.
    
    Args:
        path_str: The original file path
        base_dir: The base directory to strip from the path (for recursive processing)
        custom_dir: Custom directory to use instead of the original path
        time_offset_pattern: Optional pattern for extracting time offset number
        
    Returns:
        Tuple of (processed_filepath, time_offset)
        time_offset will be None if no number is detected or pattern is not provided
    """
    # Convert to Path objects
    path = Path(path_str)
    time_offset = None
    
    # Try to extract time offset from filename if pattern is provided
    if time_offset_pattern:
        time_offset = extract_time_offset(path.name, time_offset_pattern)
    
    # Process filepath
    if base_dir and custom_dir:
        # If we have both a base_dir and custom_dir, replace the base with custom
        base = Path(base_dir)
        try:
            # Get relative path from the base directory
            rel_path = path.relative_to(base)
            # Join with custom directory
            new_path = Path(custom_dir) / rel_path
            return str(new_path), time_offset
        except ValueError:
            # If path is not relative to base_dir, just use the filename
            return str(Path(custom_dir) / path.name), time_offset
    elif custom_dir:
        # If we only have custom_dir, just use the filename
        return str(Path(custom_dir) / path.name), time_offset
    else:
        # Otherwise return the original path
        return str(path), time_offset
    
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Publish images to Pub/Sub in Cloud Function format",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("image_path", help="Path to the image file or directory containing images")
    parser.add_argument("--project", help="Google Cloud project ID")
    parser.add_argument("--topic", help="Pub/Sub topic ID")
    parser.add_argument("--filepath", help="Custom filepath to include in the message payload (different from the actual file path)")
    parser.add_argument("--custom-dir", help="Custom directory to use as the base path in the payload (for directory processing)")
    parser.add_argument("--time-offset", type=float, help="Time offset in seconds to include in the message payload")
    parser.add_argument("--time-offset-pattern", help="Pattern to extract time offset from filename (e.g., 'frame_%05d.jpg')")
    parser.add_argument("--max-size", type=int, default=256, 
                        help="Maximum dimension for resizing (default: 256)")
    parser.add_argument("--check-auth", action="store_true", 
                        help="Check Google Cloud authentication only")
    parser.add_argument("--save-only", metavar="FILE", 
                        help="Create a test message JSON file without publishing")
    parser.add_argument("--recursive", action="store_true",
                        help="Process directories recursively")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be published without actually publishing")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed progress information for each message")
    
    args = parser.parse_args()
    
    # Check credentials first
    if args.check_auth:
        check_credentials()
        return
    
    # Check if path exists
    path = Path(args.image_path)
    if not path.exists():
        print(f"Error: Path {args.image_path} does not exist")
        sys.exit(1)
    
    # Find image files to process
    if path.is_dir() and args.recursive:
        print(f"Searching for images in {path} recursively...")
        image_files = find_image_files(path)
        if not image_files:
            print(f"No image files found in {path}")
            sys.exit(1)
        print(f"Found {len(image_files)} image files")
    elif path.is_dir():
        print(f"Error: {path} is a directory. Use --recursive to process directories")
        sys.exit(1)
    else:
        # Single file mode
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            print(f"Warning: {path} does not appear to be an image file")
        image_files = [str(path)]
    
    # Save-only mode
    if args.save_only:
        if len(image_files) > 1:
            print("Error: --save-only only works with a single image file")
            sys.exit(1)
        # Process file path and get time offset if pattern provided
        if args.filepath:
            custom_path = args.filepath
            file_time_offset = args.time_offset
        else:
            custom_path, extracted_offset = process_path(
                image_files[0], 
                None, 
                None,
                args.time_offset_pattern
            )
            file_time_offset = args.time_offset
            if extracted_offset is not None and file_time_offset is None:
                file_time_offset = extracted_offset
        
        create_test_message(image_files[0], args.max_size, args.save_only, custom_path, file_time_offset, args.verbose)
        return
    
    # Validate required args for publishing
    if not args.project or not args.topic:
        print("Error: --project and --topic are required for publishing to Pub/Sub")
        print("Use --save-only to create a test message without publishing")
        sys.exit(1)
    
    if not check_credentials():
        print("Authentication check failed. Fix credentials before continuing.")
        return
    
    # Dry run mode
    if args.dry_run:
        print("\nDRY RUN MODE - No messages will be published")
        for img_file in image_files:
            custom_path, extracted_offset = process_path(
                img_file, 
                str(path) if path.is_dir() else None, 
                args.custom_dir,
                args.time_offset_pattern
            )
            
            # Determine which time offset would be used
            file_time_offset = args.time_offset
            if extracted_offset is not None and file_time_offset is None:
                file_time_offset = extracted_offset
                
            print(f"Would publish: {img_file}")
            print(f"  with path in payload: {custom_path}")
            if file_time_offset is not None:
                print(f"  with time offset: {file_time_offset:.2f}s")
        return
    
    # Process each image
    total = len(image_files)
    success_count = 0
    
    # Create progress bar
    progress_bar = tqdm(total=total, desc="Publishing images", unit="image")
    
    for img_file in image_files:
        # Process the filepath for the message payload and extract time offset if pattern provided
        file_time_offset = args.time_offset  # Default to command line argument
        
        if args.filepath:
            # If explicit filepath is provided, use it
            custom_path = args.filepath
        else:
            # Otherwise, process the path and possibly extract time offset
            custom_path, extracted_offset = process_path(
                img_file, 
                str(path) if path.is_dir() else None, 
                args.custom_dir,
                args.time_offset_pattern
            )
            
            # Use extracted offset if available and no explicit offset provided
            if extracted_offset is not None and file_time_offset is None:
                file_time_offset = extracted_offset
        
        # Update progress bar description with time offset info if available
        status_msg = f"Processing: {Path(img_file).name}"
        if file_time_offset is not None:
            status_msg += f" (t={file_time_offset:.2f}s)"
        progress_bar.set_postfix_str(status_msg)
        
        # Publish message
        success = publish_message(
            args.project, 
            args.topic, 
            img_file, 
            args.max_size,
            custom_path,
            file_time_offset,
            args.verbose
        )
        
        if success:
            success_count += 1
        
        # Update progress bar
        progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    # Final summary
    if total > 1:
        print(f"\nSummary: Successfully published {success_count}/{total} images to Pub/Sub")
        
    # Print message about checking logs for results
    if success_count > 0:
        print("\nTo check processing status in Cloud Run logs, use:")
        print(f"gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=YOUR_SERVICE_NAME\" --project={args.project} --limit=20")
        
if __name__ == "__main__":
    main()