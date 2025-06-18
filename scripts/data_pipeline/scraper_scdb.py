#!/usr/bin/env python3
"""
SCDB (Supreme Court Database) Scraper
=====================================

Downloads the Justice-Centered Vote CSV file from the official SCDB website.
This file contains individual justice votes for each Supreme Court case.

The SCDB is the definitive source for Supreme Court case data, maintained by
Washington University in St. Louis.
"""

import os
import requests
import argparse
from pathlib import Path
import sys

# Add src to path for utils import
sys.path.append('src')
from utils.progress import tqdm, HAS_TQDM

def download_scdb_data(output_file: str = "data/raw/SCDB_2024_01_justiceCentered_Vote.csv", verbose: bool = True, quiet: bool = False):
    """
    Download the Justice-Centered Vote CSV from SCDB.
    
    This downloads the latest version of the SCDB Justice-Centered data,
    which includes individual justice votes for each case.
    """
    
    # Direct download URL for the latest Justice-Centered Vote CSV
    # Based on the SCDB website: http://scdb.wustl.edu/data.php
    scdb_url = "http://scdb.wustl.edu/_brickFiles/2024_01/SCDB_2024_01_justiceCentered_Vote.csv.zip"
    
    if not quiet:
        print(f"Downloading SCDB Justice-Centered Vote data...")
        print(f"Source: {scdb_url}")
        print(f"Output: {output_file}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Download the ZIP file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        if verbose:
            print(f"Fetching: {scdb_url}")
        
        response = requests.get(scdb_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Save the ZIP file temporarily
        zip_path = output_file + ".zip"
        
        total_size = int(response.headers.get('content-length', 0))
        if verbose:
            print(f"Downloading {total_size / (1024*1024):.1f} MB...")
        
        disable_tqdm = quiet and not HAS_TQDM
        
        with tqdm(total=total_size, desc="Downloading SCDB", disable=disable_tqdm,
                  unit="B", unit_scale=True, leave=True) as pbar:
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        if verbose:
            print(f"\nDownload complete: {zip_path}")
        
        # Extract the CSV from the ZIP
        import zipfile
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents to find the CSV file
            csv_files = [name for name in zip_ref.namelist() if name.endswith('.csv')]
            
            if not csv_files:
                raise Exception("No CSV file found in the downloaded ZIP")
            
            csv_filename = csv_files[0]  # Should be SCDB_2024_01_justiceCentered_Vote.csv
            
            if verbose:
                print(f"Extracting: {csv_filename}")
            
            # Extract the CSV file
            with zip_ref.open(csv_filename) as csv_file:
                with open(output_file, 'wb') as output:
                    output.write(csv_file.read())
        
        # Clean up the ZIP file
        os.remove(zip_path)
        
        # Check file size
        file_size = os.path.getsize(output_file)
        if not quiet:
            print(f"✅ Successfully downloaded SCDB data")
            print(f"   File: {output_file}")
            print(f"   Size: {file_size / (1024*1024):.1f} MB")
        
        # Verify the file has data
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = sum(1 for _ in f)
            if not quiet:
                print(f"   Rows: {lines:,} (including header)")
        
        return True
        
    except Exception as e:
        if not quiet:
            print(f"❌ Failed to download SCDB data: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SCDB Justice-Centered Vote data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/raw/SCDB_2024_01_justiceCentered_Vote.csv",
        help="Output CSV file path"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    success = download_scdb_data(args.output, verbose=verbose, quiet=args.quiet)
    
    if not success:
        exit(1) 