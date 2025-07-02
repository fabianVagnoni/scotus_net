#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import sys

# Add scripts to path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.progress import tqdm, HAS_TQDM

# Case disposition outcome mapping
CASE_DISPOSITION_OUTCOME = {
    1:  "in favour",   # stay, petition, or motion granted
    2:  "against",     # affirmed (petitioning party lost)
    3:  "in favour",   # reversed
    4:  "in favour",   # reversed and remanded
    5:  "in favour",   # vacated and remanded
    6:  "in favour",   # affirmed & reversed/vacated in part
    7:  "in favour",   # affirmed & reversed/vacated in part and remanded
    8:  "in favour",   # vacated
    9:  "against",     # petition denied or appeal dismissed
    10: "unclear",     # certification to/from a lower court
    11: "unclear"      # no disposition
}

def process_cases_metadata(input_file: str, output_file: str, verbose: bool = True, quiet: bool = False):
    """
    Process SCDB case metadata to extract key information and compute voting percentages.
    """
    if verbose:
        print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if verbose:
        print(f"Original data shape: {df.shape}")
        print(f"Total records: {len(df):,}")
    
    # Keep only the required columns
    required_columns = [
        'caseIssuesId',
        'usCite', 
        'chief',
        'caseName',
        'petitioner',
        'petitionerState', 
        'respondentState',
        'caseOrigin',
        'caseOriginState',
        'issueArea',
        'lawType',
        'justiceName',
        'vote',
        'caseDisposition',
        'majVotes',
        'minVotes'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        return
    
    # Filter to required columns
    df_filtered = df[required_columns].copy()
    if verbose:
        print(f"Filtered data shape: {df_filtered.shape}")
    
    # Group by case and compute voting statistics
    if not quiet:
        print("Computing voting percentages by case...")
    
    case_stats = []
    
    # Get unique cases for progress tracking
    unique_cases = df_filtered['caseIssuesId'].unique()
    disable_tqdm = quiet and not HAS_TQDM
    
    with tqdm(total=len(unique_cases), desc="Processing cases", disable=disable_tqdm,
              unit="case", leave=True) as pbar:
        
        for case_id in unique_cases:
            case_group = df_filtered[df_filtered['caseIssuesId'] == case_id]
            # Get case metadata (should be the same for all justices in the case)
            case_info = case_group.iloc[0].copy()
            
            # Get case disposition and vote counts
            case_disposition = case_info['caseDisposition']
            maj_votes = case_info['majVotes']
            min_votes = case_info['minVotes']
            total_justices = len(case_group)
            
            # Calculate percentages based on case disposition
            if pd.isna(case_disposition) or pd.isna(maj_votes) or pd.isna(min_votes):
                # Handle missing data
                pct_in_favor = pct_against = pct_absent = -1
                votes_in_favor = votes_against = votes_absent = -1
                if verbose:
                    print(f"Case {case_id} has missing disposition or vote data")
            elif case_disposition in {1, 3, 4, 5, 6, 7, 8}:
                # Petitioner wins cases
                pct_absent = (9 - maj_votes - min_votes) / 9
                pct_in_favor = maj_votes / 9
                pct_against = min_votes / 9
                votes_in_favor = int(maj_votes)
                votes_against = int(min_votes)
                votes_absent = int(9 - maj_votes - min_votes)
            elif case_disposition in {2, 9}:
                # Petitioner loses cases
                pct_absent = (9 - maj_votes - min_votes) / 9
                pct_in_favor = min_votes / 9
                pct_against = maj_votes / 9
                votes_in_favor = int(min_votes)
                votes_against = int(maj_votes)
                votes_absent = int(9 - maj_votes - min_votes)
            else:
                # Unclear cases
                pct_absent = -1
                pct_in_favor = -1
                pct_against = -1
                votes_in_favor = votes_against = votes_absent = -1

            # Create result record
            case_result = {
                'caseIssuesId': case_id,
                'usCite': case_info['usCite'],
                'chief': case_info['chief'], 
                'caseName': case_info['caseName'],
                'petitioner': case_info['petitioner'],
                'petitionerState': case_info['petitionerState'],
                'respondentState': case_info['respondentState'],
                'caseOrigin': case_info['caseOrigin'],
                'caseOriginState': case_info['caseOriginState'],
                'issueArea': case_info['issueArea'],
                'lawType': case_info['lawType'],
                'caseDisposition': case_disposition,
                'majVotes': maj_votes,
                'minVotes': min_votes,
                'total_justices_voting': total_justices,
                'votes_in_favor': votes_in_favor,
                'votes_against': votes_against, 
                'votes_absent': votes_absent,
                'pct_in_favor': pct_in_favor,
                'pct_against': pct_against,
                'pct_absent': pct_absent,
                # Also include individual justice votes as a reference
                'justice_votes': '; '.join([f"{row['justiceName']}:{row['vote']}" for _, row in case_group.iterrows()])
            }
            
            case_stats.append(case_result)
            pbar.update(1)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(case_stats)
    
    if not quiet:
        print(f"Processed {len(result_df):,} unique cases")
    
    if verbose:
        # Only calculate averages for valid cases (not -1)
        valid_cases = result_df[result_df['pct_in_favor'] != -1]
        if len(valid_cases) > 0:
            print(f"Sample voting percentages (valid cases only):")
            print(f"  Average % in favor: {valid_cases['pct_in_favor'].mean():.3f}")
            print(f"  Average % against: {valid_cases['pct_against'].mean():.3f}")
            print(f"  Average % absent: {valid_cases['pct_absent'].mean():.3f}")
        
        # Display some sample cases
        print("\nSample processed cases:")
        print("="*100)
        sample_cases = result_df.head(3)
        for _, case in sample_cases.iterrows():
            print(f"Case: {case['caseName']}")
            print(f"  ID: {case['caseIssuesId']}")
            print(f"  Citation: {case['usCite']}")
            print(f"  Chief: {case['chief']}")
            print(f"  Case Disposition: {case['caseDisposition']}")
            print(f"  Maj/Min Votes: {case['majVotes']}/{case['minVotes']}")
            if case['pct_in_favor'] != -1:
                print(f"  Votes: {case['votes_in_favor']} in favor, {case['votes_against']} against, {case['votes_absent']} absent")
                print(f"  Percentages: {case['pct_in_favor']:.1%} in favor, {case['pct_against']:.1%} against, {case['pct_absent']:.1%} absent")
            else:
                print(f"  Votes: Unable to determine (unclear disposition or missing data)")
            print("-"*80)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    
    if not quiet:
        print(f"Cases metadata saved to: {output_file}")
    
    # Additional statistics
    if verbose:
        print(f"\nDataset Statistics:")
        print(f"  Date range: {result_df['caseIssuesId'].str[:4].min()} - {result_df['caseIssuesId'].str[:4].max()}")
        print(f"  Unique chiefs: {result_df['chief'].nunique()}")
        
        valid_cases = result_df[result_df['pct_in_favor'] != -1]
        invalid_cases = result_df[result_df['pct_in_favor'] == -1]
        
        print(f"  Valid cases (clear disposition): {len(valid_cases):,}")
        print(f"  Invalid cases (unclear/missing data): {len(invalid_cases):,}")
        
        if len(valid_cases) > 0:
            print(f"  Cases with unanimous decisions (100% in favor): {(valid_cases['pct_in_favor'] == 1.0).sum()}")
            print(f"  Cases with no abstentions: {(valid_cases['pct_absent'] == 0.0).sum()}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description="Process SCDB case metadata and compute voting percentages"
    )
    parser.add_argument(
        "--input",
        "-i", 
        default="data/raw/SCDB_2024_01_justiceCentered_Vote.csv",
        help="Path to input SCDB CSV file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/cases_metadata.csv", 
        help="Path to save processed cases metadata"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    process_cases_metadata(args.input, args.output, verbose=verbose, quiet=args.quiet)

if __name__ == "__main__":
    main()







