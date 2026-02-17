#!/usr/bin/env python3
"""
Create a standalone distribution ZIP file for HF Propagation Conditions Display.

This script packages all necessary files into a distributable ZIP archive
that can be shared and run on any Linux system with Python 3.9+.

Usage:
    python create_distribution.py

Output:
    dist/hf-propagation-v1.0.0.zip
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

VERSION = "1.1.0"
DIST_NAME = f"hf-propagation-v{VERSION}"


def create_distribution():
    """Create the distribution ZIP file."""
    script_dir = Path(__file__).parent
    dist_dir = script_dir / "dist"
    package_dir = dist_dir / DIST_NAME

    print(f"Creating distribution: {DIST_NAME}")
    print(f"Target: {dist_dir / f'{DIST_NAME}.zip'}")
    print()

    # Clean previous build
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    # Create distribution structure
    package_dir.mkdir(parents=True)
    propagation_dir = package_dir / "propagation"
    propagation_dir.mkdir()

    # Python package files
    package_files = [
        "__init__.py",
        "data_client.py",
        "widgets.py",
        "main_window.py",
        "main_direct.py",
    ]

    # Copy package files
    for filename in package_files:
        src = script_dir / filename
        if src.exists():
            shutil.copy(src, propagation_dir / filename)
            print(f"  + propagation/{filename}")
        else:
            print(f"  ! Missing: {filename}")

    # Documentation files
    doc_files = [
        ("README.md", "README.md"),
        ("INSTALL.md", "INSTALL.md"),
        ("requirements.txt", "requirements.txt"),
    ]

    for src_name, dst_name in doc_files:
        src = script_dir / src_name
        if src.exists():
            shutil.copy(src, package_dir / dst_name)
            print(f"  + {dst_name}")
        else:
            print(f"  ! Missing: {src_name}")

    # Copy shell scripts
    for script in ["install.sh", "run.sh"]:
        src = script_dir / script
        if src.exists():
            dst = package_dir / script
            shutil.copy(src, dst)
            dst.chmod(0o755)
            print(f"  + {script} (executable)")

    # Create VERSION file
    version_file = package_dir / "VERSION"
    version_file.write_text(f"{VERSION}\n")
    print(f"  + VERSION")

    # Create ZIP file
    print()
    zip_path = dist_dir / f"{DIST_NAME}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(dist_dir)
                zipf.write(file_path, arcname)

    # Calculate size
    size_kb = zip_path.stat().st_size / 1024

    # Cleanup temporary directory
    shutil.rmtree(package_dir)

    print(f"Distribution created successfully!")
    print()
    print(f"  File: {zip_path}")
    print(f"  Size: {size_kb:.1f} KB")
    print()
    print("To distribute:")
    print(f"  1. Share the file: {DIST_NAME}.zip")
    print("  2. Recipients extract and run: ./install.sh && ./run.sh")
    print()

    return zip_path


if __name__ == "__main__":
    create_distribution()
