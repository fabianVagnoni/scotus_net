#!/usr/bin/env python3
"""
SCOTUS AI Data Pipeline
======================

Main pipeline orchestrator that executes the complete data processing pipeline
for Supreme Court cases and justice biographies FROM SCRATCH.

Pipeline Steps:
1. Scrape Justice Metadata (Wikipedia → data/raw/justices.json)
2. Scrape Justice Biographies (Wikipedia → data/raw/bios/)
3. Process Cases Metadata (CSV processing → data/processed/cases_metadata.csv)
4. Scrape Case Descriptions (Justia + AI filtering → data/raw/case_descriptions_ai_filtered/)
5. Process Justice Biographies (metadata enrichment → data/processed/bios/)
6. Create Case Metadata Descriptions (→ data/processed/case_metadata/)
7. Create Complete Case Descriptions (→ data/processed/case_descriptions/)
8. Build Final Case Dataset (→ data/processed/case_dataset.json)

Usage:
    python main.py                    # Run full pipeline from scratch
    python main.py --step scrape      # Run only data collection steps
    python main.py --step process     # Run only processing steps  
    python main.py --from-step 5      # Start from step 5 (assumes data collected)
    python main.py --quick            # Quick mode: reduced processing
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed. Install with: pip install tqdm")
    
    # Fallback progress bar class
    class tqdm:
        def __init__(self, total=None, desc="", unit="", **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
            print(f"🔄 {desc}")
        
        def update(self, n=1):
            self.n += n
            if self.total:
                percent = (self.n / self.total) * 100
                print(f"   Progress: {self.n}/{self.total} ({percent:.1f}%)")
            
        def set_description(self, desc):
            self.desc = desc
            
        def close(self):
            pass
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            self.close()

# Add src to path for imports
sys.path.append('src/data_pipeline')

def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_step(step_num: int, total_steps: int, description: str):
    """Print a step header."""
    print(f"\n{'📍' if step_num <= total_steps else '✅'} STEP {step_num}/{total_steps}: {description}")
    print("-" * 60)

def run_script(script_path: str, args: List[str] = None, description: str = ""):
    """Run a Python script with optional arguments - SIMPLE and FAST."""
    if args is None:
        args = []
    
    cmd = [sys.executable, script_path] + args
    print(f"🚀 Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Just run it directly - no fancy monitoring or output capturing
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f}s")
        print(f"Return code: {e.returncode}")
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Unexpected error after {elapsed:.1f}s: {e}")
        return False

def check_data_status():
    """Check what data already exists (informational only)."""
    print_header("🔍 CHECKING EXISTING DATA")
    
    data_files = [
        ("data/raw/justices.json", "Justice metadata"),
        ("data/processed/cases_metadata.csv", "Processed cases metadata"),
        ("data/raw/bios/", "Justice biographies"),
        ("data/raw/case_descriptions_ai_filtered/", "AI-filtered case descriptions"),
        ("data/processed/bios/", "Processed biographies"),
        ("data/processed/case_metadata/", "Case metadata descriptions"),
        ("data/processed/case_descriptions/", "Complete case descriptions"),
        ("data/processed/case_dataset.json", "Final ML dataset")
    ]
    
    existing_data = []
    missing_data = []
    
    for path, description in data_files:
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path) if f.endswith('.txt')])
                print(f"✅ Found: {description} ({count} files)")
                existing_data.append((path, description, count))
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✅ Found: {description} ({size_mb:.1f} MB)")
                existing_data.append((path, description, size_mb))
        else:
            print(f"⚪ Missing: {description}")
            missing_data.append((path, description))
    
    if existing_data:
        print(f"\n📊 Pipeline can start from any step based on existing data")
    else:
        print(f"\n🚀 Starting completely from scratch - will collect all data")
    
    return len(existing_data), len(missing_data)

def create_directories():
    """Create necessary directories for the entire pipeline."""
    print_header("📁 CREATING DIRECTORIES")
    
    all_dirs = [
        "data/raw",
        "data/raw/bios",
        "data/raw/case_descriptions_ai_filtered",
        "data/processed",
        "data/processed/bios",
        "data/processed/case_metadata", 
        "data/processed/case_descriptions"
    ]
    
    for dir_path in all_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")

# SCRAPING STEPS (Data Collection)

def step_scrape_justices(output_file: str = "data/raw/justices.json"):
    """Step 1: Scrape justice metadata from Wikipedia."""
    print_step(1, 8, "Scraping Justice Metadata from Wikipedia")
    
    # No arguments needed - script uses defaults
    args = []
    
    return run_script("src/data_pipeline/scraper_justices.py", args,
                     "Scraping justice information from Wikipedia")

def step_scrape_bios(justices_file: str = "data/raw/justices.json",
                    output_dir: str = "data/raw/bios"):
    """Step 2: Scrape justice biographies from Wikipedia."""
    print_step(2, 8, "Scraping Justice Biographies from Wikipedia")
    
    args = [
        "--input", justices_file,
        "--output", output_dir
    ]
    
    return run_script("src/data_pipeline/scraper_bios.py", args,
                     "Scraping justice biographies from Wikipedia")

def step_process_cases_metadata(input_file: str = "data/raw/SCDB_2024_01_justiceCentered_Vote.csv",
                               output_file: str = "data/processed/cases_metadata.csv"):
    """Step 3: Process raw cases metadata CSV."""
    print_step(3, 8, "Processing Cases Metadata")
    
    args = [
        "--input", input_file,
        "--output", output_file
    ]
    
    return run_script("src/data_pipeline/process_cases_metadata.py", args,
                     "Processing Supreme Court database CSV")

def step_scrape_case_descriptions(metadata_file: str = "data/processed/cases_metadata.csv",
                                 output_dir: str = "data/raw/case_descriptions_ai_filtered",
                                 limit: int = None):
    """Step 4: Scrape case descriptions with AI filtering."""
    print_step(4, 8, "Scraping Case Descriptions with AI Filtering")
    
    args = [
        "--input", metadata_file,
        "--output", output_dir
    ]
    
    if limit:
        args.extend(["--limit", str(limit)])
    
    return run_script("src/data_pipeline/scraper_case_descriptions.py", args,
                     "Scraping case descriptions from Justia with Gemini AI filtering")

# PROCESSING STEPS (Data Processing)

def step_process_bios(justices_metadata: str = "data/raw/justices.json",
                     input_dir: str = "data/raw/bios",
                     output_dir: str = "data/processed/bios"):
    """Step 5: Process justice biographies."""
    print_step(5, 8, "Processing Justice Biographies")
    
    args = [
        "--input", input_dir,
        "--output", output_dir,
        "--metadata", justices_metadata
    ]
    
    return run_script("src/data_pipeline/process_bios.py", args,
                     "Processing biographies with metadata enrichment")

def step_create_case_metadata(input_file: str = "data/processed/cases_metadata.csv",
                             output_dir: str = "data/processed/case_metadata"):
    """Step 6: Create case metadata descriptions."""
    print_step(6, 8, "Creating Case Metadata Descriptions")
    
    args = [
        "--input", input_file,
        "--output", output_dir
    ]
    
    return run_script("src/data_pipeline/case_metadata_creation.py", args,
                     "Creating natural language case metadata")

def step_create_case_descriptions(metadata_file: str = "data/processed/cases_metadata.csv",
                                 descriptions_dir: str = "data/raw/case_descriptions_ai_filtered",
                                 output_dir: str = "data/processed/case_descriptions"):
    """Step 7: Create complete case descriptions."""
    print_step(7, 8, "Creating Complete Case Descriptions")
    
    args = [
        "--metadata", metadata_file,
        "--descriptions", descriptions_dir,
        "--output", output_dir
    ]
    
    return run_script("src/data_pipeline/case_descriptions_creation.py", args,
                     "Combining metadata with AI descriptions")

def step_build_final_dataset(csv_file: str = "data/processed/cases_metadata.csv",
                           case_metadata_dir: str = "data/processed/case_metadata",
                           case_descriptions_dir: str = "data/processed/case_descriptions",
                           bios_dir: str = "data/processed/bios",
                           output_file: str = "data/processed/case_dataset.json"):
    """Step 8: Build final case dataset."""
    print_step(8, 8, "Building Final Case Dataset")
    
    args = [
        "--csv", csv_file,
        "--case-metadata-dir", case_metadata_dir,
        "--case-descriptions-dir", case_descriptions_dir,
        "--bios-dir", bios_dir,
        "--output", output_file
    ]
    
    return run_script("src/data_pipeline/build_case_dataset.py", args,
                     "Building JSON dataset with file paths and voting data")

def run_full_pipeline(from_step: int = 1, quick_mode: bool = False):
    """Run the complete data processing pipeline from scratch or from a specific step."""
    print_header("🚀 SCOTUS AI DATA PIPELINE", "=")
    print("Starting complete data processing pipeline from scratch...")
    
    start_time = time.time()
    
    # Check current data status (informational)
    existing_count, missing_count = check_data_status()
    
    # Create all necessary directories
    create_directories()
    
    # Define all 8 pipeline steps
    all_steps = [
        (1, "Scrape Justice Metadata", step_scrape_justices),
        (2, "Scrape Justice Biographies", step_scrape_bios),
        (3, "Process Cases Metadata", step_process_cases_metadata),
        (4, "Scrape Case Descriptions", lambda: step_scrape_case_descriptions(limit=10 if quick_mode else None)),
        (5, "Process Justice Biographies", step_process_bios),
        (6, "Create Case Metadata", step_create_case_metadata),
        (7, "Create Case Descriptions", step_create_case_descriptions),
        (8, "Build Final Dataset", step_build_final_dataset)
    ]
    
    # Filter steps based on from_step parameter
    active_steps = [step for step in all_steps if step[0] >= from_step]
    total_steps = len(active_steps)
    
    if from_step > 1:
        print(f"\n⏭️  Starting from step {from_step}, skipping {from_step-1} initial steps")
    
    # Simple pipeline progress bar
    with tqdm(total=total_steps, desc="🚀 Pipeline Progress", unit="step") as pbar:
        
        completed_steps = 0
        
        for step_num, step_name, step_func in active_steps:
            pbar.set_description(f"🔄 Step {step_num}/8: {step_name}")
            
            try:
                success = step_func()
                
                if not success:
                    pbar.set_description(f"❌ Failed at Step {step_num}")
                    print(f"\n❌ Step {step_num} ({step_name}) failed. Stopping pipeline.")
                    return False
                
                completed_steps += 1
                pbar.update(1)
                pbar.set_description(f"✅ Step {step_num}/8 Complete")
                
            except Exception as e:
                pbar.set_description(f"❌ Error at Step {step_num}")
                print(f"\n❌ Step {step_num} ({step_name}) failed with error: {e}")
                return False
        
        pbar.set_description("🎉 All Steps Completed!")
    
    # Pipeline completion
    total_time = time.time() - start_time
    
    print_header("�� PIPELINE COMPLETED SUCCESSFULLY!", "=")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🏁 Completed {completed_steps}/{total_steps} steps successfully")
    
    # Show comprehensive output summary
    output_locations = [
        ("data/raw/justices.json", "Justice metadata"),
        ("data/raw/bios/", "Raw justice biographies"),
        ("data/processed/cases_metadata.csv", "Processed cases metadata"),
        ("data/raw/case_descriptions_ai_filtered/", "AI-filtered case descriptions"),
        ("data/processed/bios/", "Processed justice biographies"),
        ("data/processed/case_metadata/", "Case metadata descriptions"),
        ("data/processed/case_descriptions/", "Complete case descriptions"),
        ("data/processed/case_dataset.json", "Final ML dataset")
    ]
    
    print(f"\n📋 Generated outputs:")
    for output_path, description in output_locations:
        if os.path.exists(output_path):
            if os.path.isdir(output_path):
                count = len([f for f in os.listdir(output_path) if f.endswith('.txt')])
                print(f"   ✅ {description}: {output_path} ({count} files)")
            else:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   ✅ {description}: {output_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ⚪ {description}: {output_path} (not created)")
    
    print(f"\n💡 Your complete SCOTUS AI dataset is ready for ML training!")
    print(f"   🎯 Main dataset file: data/processed/case_dataset.json")
    print(f"   📊 All source data preserved in data/raw/ directories")
    
    return True

def run_single_step(step: str):
    """Run a single pipeline step."""
    print_header(f"🎯 RUNNING SINGLE STEP: {step.upper()}")
    
    step_mapping = {
        "scrape-justices": step_scrape_justices,
        "scrape-bios": step_scrape_bios,
        "process-cases": step_process_cases_metadata,
        "scrape-cases": step_scrape_case_descriptions,
        "process-bios": step_process_bios,
        "case-metadata": step_create_case_metadata,
        "case-descriptions": step_create_case_descriptions,
        "dataset": step_build_final_dataset
    }
    
    if step in step_mapping:
        return step_mapping[step]()
    else:
        print(f"❌ Unknown step: {step}")
        print(f"Available steps: {', '.join(step_mapping.keys())}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="SCOTUS AI Data Pipeline - Complete data processing orchestrator FROM SCRATCH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Run full pipeline from scratch
  python main.py --from-step 5       # Start from step 5 (process biographies)
  python main.py --step scrape-bios  # Run only biography scraping
  python main.py --step dataset      # Run only final dataset creation
  python main.py --quick             # Quick mode (reduced processing)
  python main.py --check             # Only check data status
  
Complete Pipeline Steps:
  1. scrape-justices   - Scrape justice metadata from Wikipedia
  2. scrape-bios      - Scrape justice biographies from Wikipedia
  3. process-cases    - Process cases metadata CSV
  4. scrape-cases     - Scrape case descriptions with AI filtering
  5. process-bios     - Process justice biographies with metadata
  6. case-metadata    - Create case metadata descriptions
  7. case-descriptions - Create complete case descriptions
  8. dataset          - Build final JSON dataset

Requirements:
  - Python packages: tqdm (pip install tqdm)
  - Environment: GEMMA_KEY for AI filtering (in .env file)
"""
    )
    
    parser.add_argument(
        "--step",
        choices=["scrape-justices", "scrape-bios", "process-cases", "scrape-cases", 
                "process-bios", "case-metadata", "case-descriptions", "dataset"],
        help="Run a single pipeline step only"
    )
    
    parser.add_argument(
        "--from-step",
        type=int,
        choices=range(1, 9),
        default=1,
        help="Start pipeline from specific step (1-8)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true", 
        help="Quick mode - reduced processing for testing (limits case descriptions to 10)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check data status, don't run pipeline"
    )
    
    args = parser.parse_args()
    
    # Handle check mode
    if args.check:
        check_data_status()
        return True
    
    # Handle single step mode
    if args.step:
        return run_single_step(args.step)
    
    # Handle full pipeline mode
    success = run_full_pipeline(from_step=args.from_step, quick_mode=args.quick)
    
    if not success:
        print_header("❌ PIPELINE FAILED", "!")
        print("Check the error messages above and fix any issues.")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main() 