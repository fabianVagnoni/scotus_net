#!/usr/bin/env python3
"""
SCOTUS AI Data Pipeline
======================

Main pipeline orchestrator that executes the complete data processing pipeline
for Supreme Court cases and justice biographies FROM SCRATCH.

Pipeline Steps:
1. Scrape Justice Metadata (Wikipedia → data/raw/justices.json)
2. Scrape Justice Biographies (Wikipedia → data/raw/bios/)
3. Download SCDB Data (Supreme Court Database → data/raw/SCDB_2024_01_justiceCentered_Vote.csv)
4. Process Cases Metadata (CSV processing → data/processed/cases_metadata.csv)
5. Scrape Case Descriptions (Justia + AI filtering → data/raw/case_descriptions_ai_filtered/)
6. Process Justice Biographies (metadata enrichment → data/processed/bios/)
7. Create Case Metadata Descriptions (→ data/processed/case_metadata/)
8. Create Complete Case Descriptions (→ data/processed/case_descriptions/)
9. Build Final Case Dataset (→ data/processed/case_dataset.json)

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

# Add scripts to path for imports
sys.path.append('scripts/data_pipeline')

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
    """Run a Python script with optional arguments - allows real-time progress bar display."""
    if args is None:
        args = []
    
    cmd = [sys.executable, script_path] + args
    print(f"🚀 Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Don't capture output so tqdm progress bars can display in real-time
        result = subprocess.run(cmd, check=False)  # Don't raise exception on non-zero exit
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed in {elapsed:.1f}s")
            return True
        elif result.returncode == 2:
            # Special handling for API quota/rate limit interruption
            print(f"⚠️  Interrupted due to API limits after {elapsed:.1f}s")
            print(f"🛑 PIPELINE INTERRUPTED: API quota/rate limit exceeded")
            print(f"💡 The scraper has been stopped to respect API limits.")
            print(f"🔄 To resume processing later, run the same command - it will automatically continue from where it left off")
            return "interrupted"  # Special return value to indicate interruption
        else:
            print(f"❌ Failed after {elapsed:.1f}s")
            print(f"Return code: {result.returncode}")
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
                
                # Special handling for case descriptions to show completion percentage
                if "case_descriptions_ai_filtered" in path:
                    expected_total = get_expected_case_count()
                    if expected_total > 0:
                        completion_pct = (count / expected_total) * 100
                        missing_count = expected_total - count
                        if missing_count <= 150:
                            status = "✅ Complete"
                        else:
                            status = "🔄 Incomplete"
                        print(f"{status}: {description} ({count}/{expected_total} files, {completion_pct:.1f}% complete)")
                    else:
                        print(f"✅ Found: {description} ({count} files)")
                else:
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

def determine_pipeline_start_step():
    """Determine the optimal starting step based on existing data."""
    # Define step dependencies with more precise requirements
    step_requirements = [
        (1, [], "Scrape Justice Metadata"),  # Step 1: No requirements (start from scratch)
        (2, [("data/raw/justices.json", "file")], "Scrape Justice Biographies"),  # Step 2: Needs justice metadata file
        (3, [], "Download SCDB Data"),  # Step 3: Independent download
        (4, [("data/raw/SCDB_2024_01_justiceCentered_Vote.csv", "file")], "Process Cases Metadata"),  # Step 4: Needs SCDB data file
        (5, [("data/processed/cases_metadata.csv", "file")], "Scrape Case Descriptions"),  # Step 5: Needs processed cases file
        (6, [("data/raw/justices.json", "file"), ("data/raw/bios/", "dir_with_files")], "Process Justice Biographies"),  # Step 6: Needs justice metadata and raw bios
        (7, [("data/processed/cases_metadata.csv", "file")], "Create Case Metadata"),  # Step 7: Needs processed cases file
        (8, [("data/processed/cases_metadata.csv", "file"), ("data/raw/case_descriptions_ai_filtered/", "dir_with_files_complete")], "Create Complete Case Descriptions"),  # Step 8: Needs processed cases AND complete AI descriptions
        (9, [("data/processed/cases_metadata.csv", "file"), ("data/processed/case_metadata/", "dir_with_files"), ("data/processed/case_descriptions/", "dir_with_files"), ("data/processed/bios/", "dir_with_files")], "Build Final Dataset")  # Step 9: Needs all processed data
    ]
    
    # Check which steps can be started (have all requirements)
    possible_steps = []
    
    for step_num, requirements, description in step_requirements:
        can_start = True
        missing_reqs = []
        
        for req_path, req_type in requirements:
            if not os.path.exists(req_path):
                can_start = False
                missing_reqs.append(f"{req_path} (missing)")
            elif req_type == "file":
                # For files, just check existence (already checked above)
                pass
            elif req_type == "dir_with_files":
                # For directories, check if they have txt files (our standard output format)
                if os.path.isdir(req_path):
                    txt_files = [f for f in os.listdir(req_path) if f.endswith('.txt')]
                    if len(txt_files) == 0:
                        can_start = False
                        missing_reqs.append(f"{req_path} (empty - 0 files)")
                else:
                    can_start = False
                    missing_reqs.append(f"{req_path} (not a directory)")
            elif req_type == "dir_with_files_complete":
                # For case descriptions, check if processing is actually complete
                if os.path.isdir(req_path):
                    txt_files = [f for f in os.listdir(req_path) if f.endswith('.txt')]
                    file_count = len(txt_files)
                    
                    # Get expected total from source CSV
                    expected_total = get_expected_case_count()
                    
                    if file_count == 0:
                        can_start = False
                        missing_reqs.append(f"{req_path} (empty - 0 files)")
                    elif expected_total > 0 and (expected_total - file_count) > 150:
                        # Allow up to 150 missing files, but beyond that consider incomplete
                        can_start = False
                        missing_reqs.append(f"{req_path} (incomplete - {file_count}/{expected_total} files, {expected_total - file_count} missing)")
                    # If within 150 files of expected total, consider complete
                else:
                    can_start = False
                    missing_reqs.append(f"{req_path} (not a directory)")
        
        if can_start:
            possible_steps.append((step_num, description))
        # Don't break here - we want to check all steps to find the right starting point
        
    if not possible_steps:
        return 1, "No existing data found"
    
    # Return the highest step number that can be started
    highest_step = max(possible_steps, key=lambda x: x[0])
    
    # Special logic: If we can't start step 8 because case descriptions are incomplete,
    # but we can start step 5, then start from step 5
    can_start_step_8 = any(step[0] == 8 for step in possible_steps)
    can_start_step_5 = any(step[0] == 5 for step in possible_steps)
    
    if not can_start_step_8 and can_start_step_5:
        # Check if the issue is incomplete AI-filtered case descriptions
        case_desc_dir = "data/raw/case_descriptions_ai_filtered/"
        if os.path.exists(case_desc_dir):
            txt_files = [f for f in os.listdir(case_desc_dir) if f.endswith('.txt')]
            file_count = len(txt_files)
            expected_total = get_expected_case_count()
            
            if file_count == 0:
                return 5, "AI-filtered case descriptions directory is empty - need to run scraper"
            elif expected_total > 0 and (expected_total - file_count) > 150:
                return 5, f"AI-filtered case descriptions incomplete ({file_count}/{expected_total} files) - need to resume scraper"
    
    return highest_step[0], f"Can continue from: {highest_step[1]}"

def get_expected_case_count() -> int:
    """Get the expected number of unique cases from the source CSV files."""
    try:
        # First try the processed cases metadata
        if os.path.exists("data/processed/cases_metadata.csv"):
            import pandas as pd
            df = pd.read_csv("data/processed/cases_metadata.csv")
            unique_count = df['usCite'].nunique()
            return unique_count
        
        # Fallback to raw SCDB data
        elif os.path.exists("data/raw/SCDB_2024_01_justiceCentered_Vote.csv"):
            import pandas as pd
            df = pd.read_csv("data/raw/SCDB_2024_01_justiceCentered_Vote.csv")
            # Get unique citations (same logic as in scraper)
            unique_citations = df[['caseIssuesId', 'usCite', 'caseName']].drop_duplicates(subset=['usCite'])
            return len(unique_citations)
        
        return 0
        
    except Exception as e:
        print(f"Warning: Could not determine expected case count: {e}")
        return 0

def interactive_pipeline_start():
    """Check existing data and ask user whether to start from scratch or continue."""
    print_header("🔍 CHECKING EXISTING DATA STATUS")
    
    # Check what data exists
    existing_count, missing_count = check_data_status()
    
    if existing_count == 0:
        print("\n🚀 No existing data found. Starting pipeline from scratch.")
        return 1
    
    # Determine the optimal starting step
    suggested_start_step, reason = determine_pipeline_start_step()
    
    print(f"\n💡 PIPELINE RECOMMENDATION:")
    print(f"   📍 Suggested starting step: {suggested_start_step}")
    print(f"   🔍 Reason: {reason}")
    
    # Ask user what they want to do
    print(f"\n❓ WHAT WOULD YOU LIKE TO DO?")
    print(f"   1️⃣  Continue from step {suggested_start_step} (use existing data)")
    print(f"   2️⃣  Start from scratch (overwrite existing data)")
    print(f"   3️⃣  Exit (cancel pipeline)")
    
    while True:
        try:
            choice = input(f"\n👉 Enter your choice (1/2/3): ").strip()
            
            if choice == "1":
                print(f"✅ Continuing from step {suggested_start_step} using existing data")
                return suggested_start_step
            elif choice == "2":
                print("✅ Starting from scratch - will overwrite existing data")
                return 1
            elif choice == "3":
                print("👋 Pipeline cancelled by user")
                return None
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Pipeline cancelled by user")
            return None
        except EOFError:
            print("\n\n👋 Pipeline cancelled by user")
            return None

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
    print_step(1, 9, "Scraping Justice Metadata from Wikipedia")
    
    args = ["--quiet", "--output", output_file]
    
    return run_script("scripts/data_pipeline/scraper_justices.py", args,
                     "Scraping justice information from Wikipedia")

def step_scrape_bios(justices_file: str = "data/raw/justices.json",
                    output_dir: str = "data/raw/bios"):
    """Step 2: Scrape justice biographies from Wikipedia."""
    print_step(2, 9, "Scraping Justice Biographies from Wikipedia")
    
    args = [
        "--quiet",
        "--input", justices_file,
        "--output", output_dir
    ]
    
    return run_script("scripts/data_pipeline/scraper_bios.py", args,
                     "Scraping justice biographies from Wikipedia")

def step_download_scdb_data(output_file: str = "data/raw/SCDB_2024_01_justiceCentered_Vote.csv"):
    """Step 3: Download SCDB Justice-Centered Vote data."""
    print_step(3, 9, "Downloading SCDB Justice-Centered Vote Data")
    
    args = [
        "--quiet",
        "--output", output_file
    ]
    
    return run_script("scripts/data_pipeline/scraper_scdb.py", args,
                     "Downloading Supreme Court Database justice vote data")

def step_process_cases_metadata(input_file: str = "data/raw/SCDB_2024_01_justiceCentered_Vote.csv",
                               output_file: str = "data/processed/cases_metadata.csv"):
    """Step 4: Process raw cases metadata CSV."""
    print_step(4, 9, "Processing Cases Metadata")
    
    args = [
        "--quiet",
        "--input", input_file,
        "--output", output_file
    ]
    
    return run_script("scripts/data_pipeline/process_cases_metadata.py", args,
                     "Processing Supreme Court database CSV")

def step_scrape_case_descriptions(metadata_file: str = "data/processed/cases_metadata.csv",
                                 output_dir: str = "data/raw/case_descriptions_ai_filtered",
                                 limit: int = None):
    """Step 5: Scrape case descriptions with AI filtering."""
    print_step(5, 9, "Scraping Case Descriptions with AI Filtering (Resume-enabled)")
    
    args = [
        "--quiet",
        "--input", metadata_file,
        "--output", output_dir,
        "--resume"  # Enable resume functionality by default
    ]
    
    if limit:
        args.extend(["--limit", str(limit)])
    
    return run_script("scripts/data_pipeline/scraper_case_descriptions.py", args,
                     "Scraping case descriptions from Justia with Gemini AI filtering (resume-enabled)")

# PROCESSING STEPS (Data Processing)

def step_process_bios(justices_metadata: str = "data/raw/justices.json",
                     input_dir: str = "data/raw/bios",
                     output_dir: str = "data/processed/bios"):
    """Step 6: Process justice biographies."""
    print_step(6, 9, "Processing Justice Biographies")
    
    args = [
        "--quiet",
        "--input", input_dir,
        "--output", output_dir,
        "--metadata", justices_metadata
    ]
    
    return run_script("scripts/data_pipeline/process_bios.py", args,
                     "Processing biographies with metadata enrichment")

def step_create_case_metadata(input_file: str = "data/processed/cases_metadata.csv",
                             output_dir: str = "data/processed/case_metadata"):
    """Step 7: Create case metadata descriptions."""
    print_step(7, 9, "Creating Case Metadata Descriptions")
    
    args = [
        "--quiet",
        "--input", input_file,
        "--output", output_dir
    ]
    
    return run_script("scripts/data_pipeline/case_metadata_creation.py", args,
                     "Creating natural language case metadata")

def step_create_case_descriptions(metadata_file: str = "data/processed/cases_metadata.csv",
                                 descriptions_dir: str = "data/raw/case_descriptions_ai_filtered",
                                 output_dir: str = "data/processed/case_descriptions"):
    """Step 8: Create complete case descriptions."""
    print_step(8, 9, "Creating Complete Case Descriptions")
    
    args = [
        "--quiet",
        "--metadata", metadata_file,
        "--descriptions", descriptions_dir,
        "--output", output_dir
    ]
    
    return run_script("scripts/data_pipeline/case_descriptions_creation.py", args,
                     "Combining metadata with AI descriptions")

def step_build_final_dataset(csv_file: str = "data/processed/cases_metadata.csv",
                           case_metadata_dir: str = "data/processed/case_metadata",
                           case_descriptions_dir: str = "data/processed/case_descriptions",
                           bios_dir: str = "data/processed/bios",
                           output_file: str = "data/processed/case_dataset.json"):
    """Step 9: Build final case dataset."""
    print_step(9, 9, "Building Final Case Dataset")
    
    args = [
        "--quiet",
        "--csv", csv_file,
        "--case-metadata-dir", case_metadata_dir,
        "--case-descriptions-dir", case_descriptions_dir,
        "--bios-dir", bios_dir,
        "--output", output_file
    ]
    
    return run_script("scripts/data_pipeline/build_case_dataset.py", args,
                     "Building JSON dataset with file paths and voting data")

def run_full_pipeline(from_step: int = 1, quick_mode: bool = False, interactive: bool = True):
    """Run the complete data processing pipeline from scratch or from a specific step."""
    print_header("🚀 SCOTUS AI DATA PIPELINE", "=")
    
    start_time = time.time()
    
    # If interactive mode and no specific from_step provided, check existing data
    if interactive and from_step == 1:
        determined_step = interactive_pipeline_start()
        if determined_step is None:
            print("Pipeline cancelled by user.")
            return False
        from_step = determined_step
    elif not interactive:
        # Non-interactive mode: just show data status
        print("Running in non-interactive mode...")
        existing_count, missing_count = check_data_status()
    
    if from_step == 1:
        print("Starting complete data processing pipeline from scratch...")
    else:
        print(f"Starting pipeline from step {from_step}...")
    
    # Create all necessary directories
    create_directories()
    
    # Define all 9 pipeline steps
    all_steps = [
        (1, "Scrape Justice Metadata", step_scrape_justices),
        (2, "Scrape Justice Biographies", step_scrape_bios),
        (3, "Download SCDB Data", step_download_scdb_data),
        (4, "Process Cases Metadata", step_process_cases_metadata),
        (5, "Scrape Case Descriptions", lambda: step_scrape_case_descriptions(limit=10 if quick_mode else None)),
        (6, "Process Justice Biographies", step_process_bios),
        (7, "Create Case Metadata", step_create_case_metadata),
        (8, "Create Case Descriptions", step_create_case_descriptions),
        (9, "Build Final Dataset", step_build_final_dataset)
    ]
    
    # Filter steps based on from_step parameter
    active_steps = [step for step in all_steps if step[0] >= from_step]
    total_steps = len(active_steps)
    
    if from_step > 1:
        print(f"\n⏭️  Starting from step {from_step}, skipping {from_step-1} initial steps")
    
    print(f"\n🚀 Executing {total_steps} pipeline steps...")
    
    completed_steps = 0
    
    for step_num, step_name, step_func in active_steps:
        try:
            success = step_func()
            
            if success == "interrupted":
                # Special handling for API quota interruption
                print(f"\n🛑 PIPELINE INTERRUPTED at Step {step_num} ({step_name})")
                print(f"⚠️  The process was stopped due to API quota/rate limits.")
                print(f"📊 Progress: Completed {completed_steps}/{total_steps} steps before interruption")
                print(f"🔄 To resume the pipeline later, run the same command - it will automatically continue from where it left off")
                return "interrupted"
            elif not success:
                print(f"\n❌ Step {step_num} ({step_name}) failed. Stopping pipeline.")
                return False
            
            completed_steps += 1
            print(f"✅ Step {step_num}/9 completed successfully\n")
            
        except Exception as e:
            print(f"\n❌ Step {step_num} ({step_name}) failed with error: {e}")
            return False
    
    # Pipeline completion
    total_time = time.time() - start_time
    
    print_header("🎉 PIPELINE COMPLETED SUCCESSFULLY!", "=")
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

def run_multiple_steps(steps: List[str]):
    """Run multiple specific pipeline steps."""
    print_header(f"🎯 RUNNING MULTIPLE STEPS: {', '.join(steps).upper()}")
    
    step_mapping = {
        "scrape-justices": (1, "Scrape Justice Metadata", step_scrape_justices),
        "scrape-bios": (2, "Scrape Justice Biographies", step_scrape_bios),
        "download-scdb": (3, "Download SCDB Data", step_download_scdb_data),
        "process-cases": (4, "Process Cases Metadata", step_process_cases_metadata),
        "scrape-cases": (5, "Scrape Case Descriptions", step_scrape_case_descriptions),
        "process-bios": (6, "Process Justice Biographies", step_process_bios),
        "case-metadata": (7, "Create Case Metadata", step_create_case_metadata),
        "case-descriptions": (8, "Create Case Descriptions", step_create_case_descriptions),
        "dataset": (9, "Build Final Dataset", step_build_final_dataset)
    }
    
    # Validate all steps exist
    invalid_steps = [step for step in steps if step not in step_mapping]
    if invalid_steps:
        print(f"❌ Invalid steps: {', '.join(invalid_steps)}")
        print(f"Available steps: {', '.join(step_mapping.keys())}")
        return False
    
    # Sort steps by their number for logical execution order
    sorted_steps = sorted([(step_mapping[step], step) for step in steps], key=lambda x: x[0][0])
    
    print(f"\n🚀 Executing {len(sorted_steps)} pipeline steps in order...")
    
    completed_steps = 0
    start_time = time.time()
    
    for (step_num, step_name, step_func), step_key in sorted_steps:
        try:
            print_step(step_num, 9, step_name)
            success = step_func()
            
            if success == "interrupted":
                print(f"\n🛑 PIPELINE INTERRUPTED at Step {step_num} ({step_name})")
                print(f"⚠️  The process was stopped due to API quota/rate limits.")
                print(f"📊 Progress: Completed {completed_steps}/{len(sorted_steps)} steps before interruption")
                return "interrupted"
            elif not success:
                print(f"\n❌ Step {step_num} ({step_name}) failed. Stopping pipeline.")
                return False
            
            completed_steps += 1
            print(f"✅ Step {step_num}/9 completed successfully\n")
            
        except Exception as e:
            print(f"\n❌ Step {step_num} ({step_name}) failed with error: {e}")
            return False
    
    # Completion summary
    total_time = time.time() - start_time
    print_header("🎉 MULTIPLE STEPS COMPLETED SUCCESSFULLY!", "=")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🏁 Completed {completed_steps}/{len(sorted_steps)} steps successfully")
    
    return True

def run_single_step(step: str):
    """Run a single pipeline step."""
    print_header(f"🎯 RUNNING SINGLE STEP: {step.upper()}")
    
    step_mapping = {
        "scrape-justices": step_scrape_justices,
        "scrape-bios": step_scrape_bios,
        "download-scdb": step_download_scdb_data,
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

def run_preset(preset: str):
    """Run a preset combination of steps."""
    print_header(f"🎯 RUNNING PRESET: {preset.upper()}")
    
    preset_mapping = {
        "rebuild-metadata": ["process-cases", "case-metadata"],
        "rebuild-dataset": ["process-cases", "dataset"],
        "metadata-and-dataset": ["process-cases", "case-metadata", "dataset"],
        "final-processing": ["case-metadata", "case-descriptions", "dataset"],
        "scrape-data": ["scrape-justices", "scrape-bios", "download-scdb", "scrape-cases"],
        "process-all": ["process-cases", "process-bios", "case-metadata", "case-descriptions", "dataset"]
    }
    
    if preset in preset_mapping:
        steps = preset_mapping[preset]
        print(f"📋 Preset '{preset}' includes steps: {', '.join(steps)}")
        return run_multiple_steps(steps)
    else:
        print(f"❌ Unknown preset: {preset}")
        print(f"Available presets: {', '.join(preset_mapping.keys())}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="SCOTUS AI Data Pipeline - Complete data processing orchestrator FROM SCRATCH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode - check existing data and ask user
  python main.py --non-interactive        # Run full pipeline from scratch (no prompts)
  python main.py --from-step 5            # Start from step 5 (skip interactive check)
  python main.py --step scrape-bios       # Run only biography scraping
  python main.py --step dataset           # Run only final dataset creation
  python main.py --steps process-cases,dataset  # Run multiple specific steps
  python main.py --preset rebuild-dataset # Run preset combination of steps
  python main.py --quick                  # Quick mode (reduced processing)
  python main.py --check                  # Only check data status
  
Complete Pipeline Steps:
  1. scrape-justices   - Scrape justice metadata from Wikipedia
  2. scrape-bios      - Scrape justice biographies from Wikipedia
  3. download-scdb    - Download SCDB justice vote data
  4. process-cases    - Process cases metadata CSV
  5. scrape-cases     - Scrape case descriptions with AI filtering
  6. process-bios     - Process justice biographies with metadata
  7. case-metadata    - Create case metadata descriptions
  8. case-descriptions - Create complete case descriptions
  9. dataset          - Build final JSON dataset

Available Presets:
  rebuild-metadata      - process-cases + case-metadata
  rebuild-dataset       - process-cases + dataset
  metadata-and-dataset  - process-cases + case-metadata + dataset
  final-processing      - case-metadata + case-descriptions + dataset
  scrape-data          - scrape-justices + scrape-bios + download-scdb + scrape-cases
  process-all          - process-cases + process-bios + case-metadata + case-descriptions + dataset

Requirements:
  - Python packages: tqdm, openai (pip install tqdm openai)
  - Environment: OPENAI_API_KEY for AI filtering (in .env file)
"""
    )
    
    parser.add_argument(
        "--step",
        choices=["scrape-justices", "scrape-bios", "download-scdb", "process-cases", "scrape-cases", 
                "process-bios", "case-metadata", "case-descriptions", "dataset"],
        help="Run a single pipeline step only"
    )
    
    parser.add_argument(
        "--steps",
        help="Run multiple specific steps (comma-separated), e.g., 'process-cases,dataset'"
    )
    
    parser.add_argument(
        "--preset",
        choices=["rebuild-metadata", "rebuild-dataset", "metadata-and-dataset", "final-processing", 
                "scrape-data", "process-all"],
        help="Run a preset combination of steps"
    )
    
    parser.add_argument(
        "--from-step",
        type=int,
        choices=range(1, 10),
        default=1,
        help="Start pipeline from specific step (1-9)"
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
    
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (don't prompt user for choices)"
    )
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    exclusive_args = [args.step, args.steps, args.preset, args.from_step != 1]
    if sum(bool(arg) for arg in exclusive_args) > 1:
        print("❌ Error: Cannot use --step, --steps, --preset, or --from-step together")
        print("Choose one approach:")
        print("  --step STEP_NAME        : Run single step")
        print("  --steps STEP1,STEP2     : Run multiple specific steps")
        print("  --preset PRESET_NAME    : Run preset combination")
        print("  --from-step N           : Run full pipeline from step N")
        return False
    
    # Handle check mode
    if args.check:
        check_data_status()
        return True
    
    # Handle single step mode
    if args.step:
        return run_single_step(args.step)
    
    # Handle multiple steps mode
    if args.steps:
        steps_list = [step.strip() for step in args.steps.split(',')]
        return run_multiple_steps(steps_list)
    
    # Handle preset mode
    if args.preset:
        return run_preset(args.preset)
    
    # Handle full pipeline mode
    interactive_mode = not args.non_interactive
    success = run_full_pipeline(from_step=args.from_step, quick_mode=args.quick, interactive=interactive_mode)
    
    if success == "interrupted":
        print_header("⚠️  PIPELINE INTERRUPTED", "!")
        print("The pipeline was stopped due to API quota/rate limits.")
        print("Run the same command later to resume processing from where it left off.")
        sys.exit(2)  # Exit code 2 for interruption
    elif not success:
        print_header("❌ PIPELINE FAILED", "!")
        print("Check the error messages above and fix any issues.")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main() 