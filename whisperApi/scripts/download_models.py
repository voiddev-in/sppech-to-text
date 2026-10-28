#!/usr/bin/env python3
"""
Downloads official Whisper checkpoints (MIT license) into the models directory.
Can be executed as:
    python scripts/download_models.py [model_name ...]

If no models are provided as CLI arguments, the models listed in
the WHISPER_MODELS environment variable (or 'tiny') are downloaded.
"""

import os
import sys
from faster_whisper import download_model

DEFAULT_MODELS_DIR = os.environ.get("WHISPER_MODELS_DIR", "./models")


def main():
    if len(sys.argv) > 1:
        models_to_download = [m.strip() for m in sys.argv[1:] if m.strip()]
    else:
        env_models = os.environ.get("WHISPER_MODELS", "tiny")
        models_to_download = [m.strip() for m in env_models.split(",") if m.strip()]

    output_base_dir = os.path.abspath(DEFAULT_MODELS_DIR)
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Downloading Whisper models to: {output_base_dir}")
    print(f"Target models: {', '.join(models_to_download)}")
    print("=" * 60)

    for model_name in models_to_download:
        target_dir = os.path.join(output_base_dir, model_name)
        print(f"--> Downloading / verifying model '{model_name}'...")
        try:
            downloaded_path = download_model(model_name, output_dir=target_dir)
            print(f"    [OK] Model '{model_name}' ready at: {downloaded_path}")
        except Exception as e:
            print(f"    [ERROR] Failed to download model '{model_name}': {e}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("All models downloaded successfully.")


if __name__ == "__main__":
    main()
