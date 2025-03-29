#!/usr/bin/env python3
"""
Script to check all aspects of image processing status at once.
"""
import argparse
import sys
import os
import importlib.util
from pathlib import Path

def load_script(script_path):
    """Load a Python script as a module."""
    spec = importlib.util.spec_from_file_location("module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check all aspects of image processing")
    parser.add_argument("filepath", help="Path to the image file to check")
    parser.add_argument("--hours", type=int, default=1, help="Number of hours to look back in logs")
    
    args = parser.parse_args()
    
    # Check if file exists
    path = Path(args.filepath)
    if not path.exists() or not path.is_file():
        print(f"Error: File {args.filepath} does not exist or is not a file")
        sys.exit(1)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Load and run each script in sequence
    scripts = [
        ("check_logs.py", "Cloud Run Logs"),
        ("check_gcs.py", "Google Cloud Storage"),
        ("check_qdrant.py", "Qdrant Vector Database")
    ]
    
    for script_name, check_name in scripts:
        script_path = script_dir / script_name
        
        if not script_path.exists():
            print(f"⚠️ Warning: Script {script_name} not found at {script_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"🔍 Checking {check_name}...")
        print(f"{'='*80}\n")
        
        try:
            # Import and run each script
            module = load_script(script_path)
            
            # Call the main function directly for logs script to pass hours parameter
            if script_name == "check_logs.py":
                # Set sys.argv manually to pass arguments to the script
                old_argv = sys.argv
                sys.argv = [script_path, args.filepath, "--hours", str(args.hours)]
                try:
                    module.main()
                finally:
                    sys.argv = old_argv
            else:
                # For other scripts, just call with the filepath
                # Set sys.argv manually to pass arguments to the script
                old_argv = sys.argv
                sys.argv = [script_path, args.filepath]
                try:
                    module.main()
                finally:
                    sys.argv = old_argv
            
        except Exception as e:
            print(f"Error running {script_name}: {e}")
    
    print(f"\n{'='*80}")
    print(" All checks completed! ")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()