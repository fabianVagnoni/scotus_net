#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os

# Petitioner code mappings (abbreviated - key ones)
PETITIONER_CODES = {
    1: "attorney general of the United States",
    27: "United States",
    28: "State",
    100: "person accused, indicted, or suspected of crime",
    111: "attorney or law firm",
    113: "bank, savings and loan, credit union, investment company",
    122: "business, corporation",
    126: "person convicted of crime",
    130: "religious organization, institution, or person",
    137: "defendant",
    145: "employee or job applicant",
    151: "employer",
    170: "Indian, including Indian tribe or nation",
    171: "insurance company",
    214: "private person",
    215: "prisoner, inmate of penal institution",
    222: "racial or ethnic minority",
    249: "union, labor organization",
    327: "Department of Justice or Attorney General",
    332: "Equal Employment Opportunity Commission",
    333: "Environmental Protection Agency",
    357: "Federal Trade Commission",
    501: "Unidentifiable"
}

# State code mappings
STATE_CODES = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    7: "Colorado", 8: "Connecticut", 9: "Delaware", 10: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 25: "Maryland", 26: "Massachusetts", 27: "Michigan", 28: "Minnesota",
    29: "Mississippi", 30: "Missouri", 31: "Montana", 32: "Nebraska", 33: "Nevada",
    34: "New Hampshire", 35: "New Jersey", 36: "New Mexico", 37: "New York",
    38: "North Carolina", 39: "North Dakota", 41: "Ohio", 42: "Oklahoma",
    43: "Oregon", 45: "Pennsylvania", 47: "Rhode Island", 48: "South Carolina",
    49: "South Dakota", 50: "Tennessee", 51: "Texas", 52: "Utah", 53: "Vermont",
    55: "Virginia", 56: "Washington", 57: "West Virginia", 58: "Wisconsin",
    59: "Wyoming", 60: "United States"
}

# Case origin codes (key ones)
CASE_ORIGIN_CODES = {
    21: "U.S. Court of Appeals, First Circuit", 22: "U.S. Court of Appeals, Second Circuit",
    23: "U.S. Court of Appeals, Third Circuit", 24: "U.S. Court of Appeals, Fourth Circuit",
    25: "U.S. Court of Appeals, Fifth Circuit", 26: "U.S. Court of Appeals, Sixth Circuit",
    27: "U.S. Court of Appeals, Seventh Circuit", 28: "U.S. Court of Appeals, Eighth Circuit",
    29: "U.S. Court of Appeals, Ninth Circuit", 30: "U.S. Court of Appeals, Tenth Circuit",
    31: "U.S. Court of Appeals, Eleventh Circuit", 32: "U.S. Court of Appeals, D.C. Circuit",
    300: "State Supreme Court", 301: "State Appellate Court", 302: "State Trial Court"
}

# Issue area codes
ISSUE_AREA_CODES = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment", 4: "Due Process",
    5: "Privacy", 6: "Attorneys", 7: "Unions", 8: "Economic Activity",
    9: "Judicial Power", 10: "Federalism", 11: "Interstate Relations", 12: "Federal Taxation",
    13: "Miscellaneous", 14: "Private Action"
}

def decode_value(value, code_dict, default="Unknown"):
    """Decode a numeric value using the provided code dictionary."""
    if pd.isna(value):
        return "Not specified"
    try:
        return code_dict.get(int(float(value)), default)
    except (ValueError, TypeError):
        return default

def create_case_description(row):
    """Generate a natural language description of a Supreme Court case."""
    
    # Basic case information
    case_name = row.get('caseName', 'Unknown Case')
    citation = row.get('usCite', 'No citation')
    chief = row.get('chief', 'Unknown Chief')
    
    # Decode parties
    petitioner = decode_value(row.get('petitioner'), PETITIONER_CODES, "Unknown petitioner")
    petitioner_state = decode_value(row.get('petitionerState'), STATE_CODES, "Unknown state")
    respondent_state = decode_value(row.get('respondentState'), STATE_CODES, "Unknown state")
    
    # Decode case characteristics
    case_origin = decode_value(row.get('caseOrigin'), CASE_ORIGIN_CODES, "Unknown court")
    case_origin_state = decode_value(row.get('caseOriginState'), STATE_CODES, "Unknown state")
    issue_area = decode_value(row.get('issueArea'), ISSUE_AREA_CODES, "Unknown issue area")
    
    # Voting information
    votes_favor = int(row.get('votes_in_favor', 0))
    votes_against = int(row.get('votes_against', 0))
    votes_absent = int(row.get('votes_absent', 0))
    pct_favor = row.get('pct_in_favor', 0) * 100
    pct_against = row.get('pct_against', 0) * 100
    
    # Determine case outcome
    if votes_favor > votes_against:
        outcome = "ruled in favor of the petitioner"
        margin = "unanimous" if votes_against == 0 else f"{votes_favor}-{votes_against}"
    elif votes_against > votes_favor:
        outcome = "ruled against the petitioner"
        margin = f"{votes_against}-{votes_favor}"
    else:
        outcome = "resulted in a tie"
        margin = f"{votes_favor}-{votes_against}"
    
    # Create the description
    description = f"""Case: {case_name} ({citation})

This Supreme Court case was decided under Chief Justice {chief}. The petitioner was {petitioner}"""
    
    if petitioner_state != "Not specified" and petitioner_state != "Unknown state":
        description += f" from {petitioner_state}"
    
    if respondent_state != "Not specified" and respondent_state != "Unknown state":
        description += f". The respondent was from {respondent_state}"
    
    description += f"""

The case originated from {case_origin}"""
    
    if case_origin_state != "Not specified" and case_origin_state != "Unknown state":
        description += f" in {case_origin_state}"
    
    description += f""". The primary issue area was {issue_area}."""
    
    return description

def sanitize_filename(case_id: str) -> str:
    """Convert case ID to a valid filename."""
    # Replace problematic characters with underscores
    import re
    filename = re.sub(r'[^\w\-.]', '_', case_id)
    return filename + ".txt"

def process_case_descriptions(input_file: str, output_dir: str, verbose: bool = True, quiet: bool = False):
    """Process cases metadata and create individual text files for each case description."""
    
    if verbose:
        print(f"Loading cases metadata from {input_file}...")
    df = pd.read_csv(input_file)
    
    if not quiet:
        print(f"Processing {len(df):,} cases...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    sample_descriptions = []
    
    for idx, row in df.iterrows():
        try:
            case_id = row['caseIssuesId']
            description = create_case_description(row)
            
            # Create filename from case ID
            filename = sanitize_filename(str(case_id))
            filepath = os.path.join(output_dir, filename)
            
            # Save description to individual file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(description)
            
            successful += 1
            
            # Keep first 2 descriptions for sample output
            if len(sample_descriptions) < 2 and verbose:
                sample_descriptions.append({
                    'caseName': row.get('caseName', ''),
                    'filename': filename,
                    'description': description
                })
            
            if (idx + 1) % 1000 == 0 and verbose:
                print(f"  Processed {idx + 1:,} cases...")
                
        except Exception as e:
            if verbose:
                print(f"Error processing case {row.get('caseIssuesId', 'unknown')}: {e}")
            failed += 1
            continue
    
    if not quiet:
        print(f"Created {successful}/{len(df)} case metadata descriptions ({successful/len(df)*100:.1f}% success rate)")
    
    if verbose:
        print(f"\n=== DETAILED SUMMARY ===")
        print(f"Total cases: {len(df):,}")
        print(f"Successful: {successful:,}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(df)*100:.1f}%")
        print(f"Case descriptions saved to directory: {output_dir}")
        
        # Show sample descriptions
        if sample_descriptions:
            print("\nSample case descriptions:")
            print("=" * 100)
            for i, sample in enumerate(sample_descriptions):
                print(f"\n{i+1}. {sample['caseName']} → {sample['filename']}")
                print("-" * 80)
                print(sample['description'])
                print()
    
    return successful

def main():
    parser = argparse.ArgumentParser(
        description="Create natural language descriptions of Supreme Court cases"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/processed/cases_metadata.csv",
        help="Path to input cases metadata CSV file"
    )
    parser.add_argument(
        "--output", 
        "-o",
        default="data/processed/case_metadata",
        help="Directory to save individual case description files"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    process_case_descriptions(args.input, args.output, verbose=verbose, quiet=args.quiet)

if __name__ == "__main__":
    main() 