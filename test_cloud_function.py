import argparse
import base64
import json
import os
import sys
from pathlib import Path

def create_test_message(image_path):
    """Create a test Pub/Sub message with the image."""
    # Check if file exists
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        print(f"Error: File {image_path} does not exist or is not a file")
        sys.exit(1)
    
    # Read image data
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Encode image data
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    # Create message
    message = {
        "filepath": str(path),
        "timestamp": "2023-01-01T12:00:00Z",
        "image": image_base64
    }
    
    # Encode message as base64 (simulating Pub/Sub)
    message_json = json.dumps(message)
    message_base64 = base64.b64encode(message_json.encode("utf-8")).decode("utf-8")
    
    # Create cloud event
    cloud_event = {
        "message": {
            "data": message_base64
        },
        "subscription": "projects/test-project/subscriptions/test-sub"
    }
    
    return cloud_event

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create a test message for the cloud function")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Create test message
    cloud_event = create_test_message(args.image_path)
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(cloud_event, f, indent=2)
        print(f"Test message written to {args.output}")
    else:
        print(json.dumps(cloud_event, indent=2))
    
    # Instructions
    print("\nTo test locally, run:")
    print(f"functions-framework --target=process_pubsub_message --signature-type=cloudevent")
    print("\nThen, in another terminal:")
    if args.output:
        print(f"curl localhost:8080 -X POST -H \"Content-Type: application/json\" -d @{args.output}")
    else:
        print("curl localhost:8080 -X POST -H \"Content-Type: application/json\" -d '[paste-json-here]'")

if __name__ == "__main__":
    main()