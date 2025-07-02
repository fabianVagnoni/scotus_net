#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import sys

# Add scripts to path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.progress import tqdm, HAS_TQDM

def load_justices_metadata(metadata_file: str) -> Dict:
    """
    Load justice metadata from JSON file.
    """
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load justices metadata from {metadata_file}: {e}")
        return {}

def get_state_mapping() -> Dict[str, str]:
    """
    Return mapping of state abbreviations to full state names.
    """
    return {
        'AL': 'Alabama',
        'AK': 'Alaska',
        'AZ': 'Arizona',
        'AR': 'Arkansas',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'IA': 'Iowa',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'ME': 'Maine',
        'MD': 'Maryland',
        'MA': 'Massachusetts',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MS': 'Mississippi',
        'MO': 'Missouri',
        'MT': 'Montana',
        'NE': 'Nebraska',
        'NV': 'Nevada',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NY': 'New York',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VT': 'Vermont',
        'VA': 'Virginia',
        'WA': 'Washington',
        'WV': 'West Virginia',
        'WI': 'Wisconsin',
        'WY': 'Wyoming',
        'DC': 'District of Columbia'
    }

def extract_birth_year(birth_death: str) -> Optional[str]:
    """
    Extract birth year from birth_death string like "(1745–1829)" or "(born 1955)"
    """
    if not birth_death:
        return None
    
    # Handle patterns like "(1745–1829)" or "(born 1955)" or "(1939–2025)"
    import re
    if "born" in birth_death:
        match = re.search(r'born (\d{4})', birth_death)
        if match:
            return match.group(1)
    else:
        match = re.search(r'\((\d{4})', birth_death)
        if match:
            return match.group(1)
    
    return None

def create_justice_metadata_header(justice_name: str, metadata: Dict) -> str:
    """
    Create a metadata header for a justice biography using only safe fields.
    """
    if justice_name not in metadata:
        return ""
    
    justice_data = metadata[justice_name]
    
    # Extract safe metadata fields (avoiding data leakage)
    name = justice_data.get('name', justice_name)
    state_abbrev = justice_data.get('state', '')
    birth_death = justice_data.get('birth_death', '')
    birth_year = extract_birth_year(birth_death)
    position = justice_data.get('position', '')
    appointment_date = justice_data.get('appointment_date', '')
    appointment_method = justice_data.get('appointment_method', '')
    nominated_by = justice_data.get('nominated_by', '')
    
    # Convert state abbreviation to full name
    state_mapping = get_state_mapping()
    state_full = state_mapping.get(state_abbrev, state_abbrev) if state_abbrev else ''
    
    # Build metadata header
    header_lines = []
    header_lines.append(f"Justice: {name}")
    
    if state_full:
        header_lines.append(f"State: {state_full}")
    
    if birth_year:
        header_lines.append(f"Birth Year: {birth_year}")
    
    if position:
        header_lines.append(f"Position: {position}")
    
    if appointment_date:
        # Clean up appointment date (remove vote counts and special characters)
        clean_date = re.sub(r'\s*\([^)]*\)\s*', '', appointment_date)
        clean_date = re.sub(r'\s*\[[^\]]*\]\s*', '', clean_date)
        clean_date = clean_date.strip()
        if clean_date:
            header_lines.append(f"Appointment Date: {clean_date}")
    
    if appointment_method and appointment_method.strip():
        header_lines.append(f"Appointment Method: {appointment_method}")
    
    if nominated_by and nominated_by.strip():
        header_lines.append(f"Nominated By: {nominated_by}")
    
    if header_lines:
        return "\n".join(header_lines) + "\n\n"
    
    return ""

def find_first_header(text: str) -> int:
    """
    Find the position where the first header starts.
    Returns the position, or -1 if no header is found.
    """
    # Pattern for any header in format === Text ===
    header_pattern = r'^=== .* ===$'
    
    match = re.search(header_pattern, text, re.MULTILINE)
    if match:
        return match.start()
    
    return -1

def find_scotus_section(text: str) -> int:
    """
    Find the position where the first SCOTUS-related section starts.
    Returns the position, or -1 if no SCOTUS section is found.
    """
    # Common patterns for SCOTUS-related sections
    scotus_patterns = [
        r'=== .*Supreme Court.*===',
        r'=== .*SCOTUS.*===',
        r'=== .*U\.S\. Supreme Court.*===',
        r'=== .*Supreme Court justice.*===',
        r'=== .*Supreme Court nomination.*===',
        r'=== .*Supreme Court tenure.*===',
        r'=== .*Supreme Court service.*===',
        r'=== .*Supreme Court and.*===',
        r'=== .*Associate Justice.*===',
        r'=== .*Chief Justice.*===',
        r'=== .*Justice of the Supreme Court.*===',
    ]
    
    earliest_match = -1
    
    for pattern in scotus_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            match_pos = matches[0].start()
            if earliest_match == -1 or match_pos < earliest_match:
                earliest_match = match_pos
    
    return earliest_match

def process_biography(input_path: str, output_path: str, justices_metadata: Dict = None) -> dict:
    """
    Process a single biography file by removing introductory paragraph and SCOTUS-related content,
    and adding justice metadata header.
    Returns stats about the processing.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    original_words = len(original_text.split())
    original_chars = len(original_text)
    
    # First, remove the introductory paragraph (everything before the first header)
    first_header_pos = find_first_header(original_text)
    intro_removed = False
    
    if first_header_pos != -1:
        # Remove everything before the first header
        text_after_intro = original_text[first_header_pos:].strip()
        intro_removed = True
    else:
        # No headers found, keep the original text
        text_after_intro = original_text
    
    # Then, find where SCOTUS content starts in the remaining text
    scotus_start = find_scotus_section(text_after_intro)
    
    if scotus_start == -1:
        # No SCOTUS section found, keep the text after intro removal
        processed_text = text_after_intro
        scotus_truncated = False
    else:
        # Truncate at the SCOTUS section
        processed_text = text_after_intro[:scotus_start].strip()
        scotus_truncated = True
    
    # Clean up any trailing whitespace and normalize line endings
    processed_text = re.sub(r'\s+$', '', processed_text, flags=re.MULTILINE)
    processed_text = processed_text.strip()
    
    # Add justice metadata header if available
    if justices_metadata:
        # Extract justice name from filename (e.g., "John_Jay.txt" -> "John Jay")
        filename = os.path.basename(input_path)
        justice_name = filename.replace('.txt', '').replace('_', ' ')
        
        # Create metadata header
        metadata_header = create_justice_metadata_header(justice_name, justices_metadata)
        
        if metadata_header:
            processed_text = metadata_header + processed_text
    
    # Save the processed text
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    processed_words = len(processed_text.split())
    processed_chars = len(processed_text)
    
    return {
        'original_words': original_words,
        'original_chars': original_chars,
        'processed_words': processed_words,
        'processed_chars': processed_chars,
        'truncated': intro_removed or scotus_truncated,
        'intro_removed': intro_removed,
        'scotus_truncated': scotus_truncated,
        'reduction_words': original_words - processed_words,
        'reduction_chars': original_chars - processed_chars,
    }

def main(input_dir: str, output_dir: str, metadata_file: str = None, verbose: bool = True, quiet: bool = False):
    """
    Process all biography files in the input directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Load justices metadata if provided
    justices_metadata = {}
    if metadata_file and os.path.exists(metadata_file):
        if verbose:
            print(f"Loading justices metadata from {metadata_file}...")
        justices_metadata = load_justices_metadata(metadata_file)
        if verbose:
            print(f"Loaded metadata for {len(justices_metadata)} justices")
    elif metadata_file:
        if verbose:
            print(f"Warning: Metadata file not found: {metadata_file}")
    else:
        if verbose:
            print("No metadata file provided - processing without enrichment")
    
    # Get all .txt files in the input directory
    bio_files = list(input_path.glob("*.txt"))
    
    if not bio_files:
        print(f"No .txt files found in '{input_dir}'")
        return
    
    if not quiet:
        print(f"Processing {len(bio_files)} biography files...")
    
    if verbose:
        print("=" * 80)
    
    total_stats = {
        'processed': 0,
        'truncated': 0,
        'total_original_words': 0,
        'total_processed_words': 0,
        'total_reduction_words': 0,
    }
    
    # Set up progress bar
    disable_tqdm = quiet and not HAS_TQDM
    
    with tqdm(bio_files, desc="Processing biographies", disable=disable_tqdm,
              unit="bio", leave=True) as pbar:
        
        for bio_file in pbar:
            justice_name = bio_file.stem  # filename without extension
            output_file = output_path / bio_file.name
            
            # Update progress bar description
            pbar.set_description(f"Processing {justice_name}")
            
            try:
                stats = process_biography(str(bio_file), str(output_file), justices_metadata)
                
                # Create status indicator based on what was removed
                status_parts = []
                if stats['intro_removed']:
                    status_parts.append("INTRO")
                if stats['scotus_truncated']:
                    status_parts.append("SCOTUS")
                
                if status_parts:
                    status = "REMOVED " + "+".join(status_parts)
                else:
                    status = "KEPT FULL"
                
                reduction_pct = (stats['reduction_words'] / stats['original_words'] * 100) if stats['original_words'] > 0 else 0
                
                if verbose:
                    tqdm.write(f"[{status:15}] {justice_name}")
                    tqdm.write(f"                {stats['original_words']} → {stats['processed_words']} words "
                              f"(-{stats['reduction_words']}, -{reduction_pct:.1f}%)")
                
                # Update totals
                total_stats['processed'] += 1
                if stats['truncated']:
                    total_stats['truncated'] += 1
                total_stats['total_original_words'] += stats['original_words']
                total_stats['total_processed_words'] += stats['processed_words']
                total_stats['total_reduction_words'] += stats['reduction_words']
                
            except Exception as e:
                if verbose:
                    tqdm.write(f"[ERROR          ] {justice_name}: {e}")
    
    if not quiet:
        print(f"Processed {total_stats['processed']} biographies ({total_stats['truncated']} truncated)")
    
    if verbose:
        print("=" * 80)
        print(f"DETAILED SUMMARY:")
        print(f"  Files processed: {total_stats['processed']}")
        print(f"  Files truncated: {total_stats['truncated']}")
        print(f"  Files kept full: {total_stats['processed'] - total_stats['truncated']}")
        
        if total_stats['total_original_words'] > 0:
            overall_reduction = (total_stats['total_reduction_words'] / total_stats['total_original_words'] * 100)
            print(f"  Total words: {total_stats['total_original_words']:,} → {total_stats['total_processed_words']:,}")
            print(f"  Overall reduction: {total_stats['total_reduction_words']:,} words ({overall_reduction:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process justice biographies by enriching with metadata and removing introductory paragraphs and SCOTUS-related content"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/raw/bios",
        help="Directory containing biography .txt files"
    )
    parser.add_argument(
        "--output",
        "-o", 
        default="data/processed/bios",
        help="Directory to save processed biographies"
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default="data/raw/justices.json",
        help="Path to justices metadata JSON file"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    main(args.input, args.output, args.metadata, verbose=verbose, quiet=args.quiet) 