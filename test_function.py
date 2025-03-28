import functions_framework
import json
import os

@functions_framework.cloud_event
def process_pubsub_message(cloud_event):
    """Simple test function that returns environment variables and basic info."""
    # Get environment variables
    env_vars = {key: value for key, value in os.environ.items()}
    
    # Basic response
    response = {
        "status": "success",
        "message": "Test function is working",
        "environment": env_vars,
        "cloud_event": {
            "id": getattr(cloud_event, "id", "unknown"),
            "type": getattr(cloud_event, "type", "unknown"),
            "source": getattr(cloud_event, "source", "unknown")
        }
    }
    
    return response