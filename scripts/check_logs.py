#!/usr/bin/env python3
"""
Script to check Cloud Run logs for image processing status.
"""
import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    from google.cloud import logging
except ImportError:
    print("Error: google-cloud-logging is not installed. Please run: pip install google-cloud-logging")
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
    project_id = os.getenv("GCP_PROJECT_ID")
    service_name = os.getenv("SERVICE_NAME")
    
    if not project_id or not service_name:
        print("Error: Missing environment variables")
        print("Make sure the .env file exists and contains the following variables:")
        print("GCP_PROJECT_ID=your-project-id")
        print("SERVICE_NAME=your-service-name")
        sys.exit(1)
    
    return project_id, service_name

def check_logs(filepath, project_id, service_name, hours=1):
    """Check Cloud Run logs for image processing status."""
    path = Path(filepath)
    filename = path.name
    
    print(f"Checking logs for processing of: {filename}")
    print(f"Project: {project_id}")
    print(f"Service: {service_name}")
    
    # Initialize client
    logging_client = logging.Client(project=project_id)
    
    # Create a filter for the logs
    filter_str = (
        f'resource.type="cloud_run_revision" '
        f'resource.labels.service_name="{service_name}" '
        f'severity>=INFO '
    )
    
    # Add time filter
    now = datetime.utcnow()
    time_ago = now - timedelta(hours=hours)
    time_filter = f'timestamp>="{time_ago.isoformat()}Z"'
    filter_str = f"{filter_str} {time_filter}"
    
    print(f"Looking at logs from the past {hours} hour(s)")
    print("Searching...")
    
    # Get logs
    entries = logging_client.list_entries(
        filter_=filter_str,
        order_by="timestamp desc",
        page_size=100
    )
    
    # Compile regex pattern to extract filename from paths
    filename_pattern = re.compile(r'(?:^|/)([^/]+\.[a-zA-Z0-9]+)$')
    
    # Process logs
    found_entries = []
    for entry in entries:
        payload = entry.payload
        if isinstance(payload, str):
            # Extract filename from logs if possible
            match = filename_pattern.search(payload)
            if match:
                log_filename = match.group(1)
                if filename in log_filename or log_filename in filename:
                    found_entries.append(entry)
            # Also check for direct matches
            elif filename in payload:
                found_entries.append(entry)
    
    # Display results
    if found_entries:
        print(f"\nFound {len(found_entries)} log entries related to this image:")
        for i, entry in enumerate(found_entries, 1):
            timestamp = entry.timestamp.isoformat()
            severity = entry.severity
            message = entry.payload
            
            print(f"\n{i}. {timestamp} [{severity}]")
            print(f"   {message}")
        
        # Check for success and errors
        success_entries = [e for e in found_entries if "Successfully processed" in e.payload]
        error_entries = [e for e in found_entries if "Error" in e.payload or "ERROR" in e.payload]
        
        if success_entries:
            print("\n✅ Success entries found! The image was likely processed successfully.")
        
        if error_entries:
            print("\n⚠️ Error entries found! The image processing may have encountered issues.")
        
        return True
    else:
        print("\nNo log entries found for this image.")
        print("This could mean:")
        print("1. The image hasn't been processed yet")
        print("2. The logs are outside the time range (try increasing --hours)")
        print("3. The log entries don't contain the exact filename")
        
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check Cloud Run logs for image processing status")
    parser.add_argument("filepath", help="Path to the image file to check")
    parser.add_argument("--hours", type=int, default=1, help="Number of hours to look back in logs (default: 1)")
    
    args = parser.parse_args()
    
    # Check if file exists
    path = Path(args.filepath)
    if not path.exists() or not path.is_file():
        print(f"Error: File {args.filepath} does not exist or is not a file")
        sys.exit(1)
    
    # Load environment variables
    project_id, service_name = load_env()
    
    # Check logs
    check_logs(str(path), project_id, service_name, args.hours)

if __name__ == "__main__":
    main()