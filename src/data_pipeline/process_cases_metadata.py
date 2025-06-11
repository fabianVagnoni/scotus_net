#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os

def process_cases_metadata(input_file: str, output_file: str):
    """
    Process SCDB case metadata to extract key information and compute voting percentages.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
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
        'vote'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
        return
    
    # Filter to required columns
    df_filtered = df[required_columns].copy()
    print(f"Filtered data shape: {df_filtered.shape}")
    
    # Group by case and compute voting statistics
    print("Computing voting percentages by case...")
    
    case_stats = []
    
    for case_id, case_group in df_filtered.groupby('caseIssuesId'):
        # Get case metadata (should be the same for all justices in the case)
        case_info = case_group.iloc[0].copy()
        
        # Count votes
        vote_counts = case_group['vote'].value_counts()
        total_justices = len(case_group)
        
        # Compute percentages (vote codes: 1=majority/in favor, 2=dissent/against, 8=recused/absent)
        in_favor_count = vote_counts.get(1.0, 0)  # majority votes
        against_count = vote_counts.get(2.0, 0)   # dissent votes  
        absent_count = vote_counts.get(8.0, 0)    # recused/absent votes
        other_count = sum(vote_counts.get(k, 0) for k in vote_counts if k not in [1.0, 2.0, 8.0])
        total_votes = in_favor_count + against_count + absent_count + other_count
        
        # Calculate percentages out of 9 justices (typical Supreme Court size)
        if total_votes == 0:
            pct_in_favor = pct_against = pct_absent = pct_other = 0.0
            print(f"Case {case_id} has no votes")
        else:
            pct_in_favor = in_favor_count / total_votes
            pct_against = against_count / total_votes
            pct_absent = absent_count / total_votes
            pct_other = other_count / total_votes

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
            'total_justices_voting': total_justices,
            'votes_in_favor': in_favor_count,
            'votes_against': against_count, 
            'votes_absent': absent_count,
            'votes_other': other_count,
            'pct_in_favor': pct_in_favor,
            'pct_against': pct_against,
            'pct_absent': pct_absent,
            'pct_other': pct_other,
            # Also include individual justice votes as a reference
            'justice_votes': '; '.join([f"{row['justiceName']}:{row['vote']}" for _, row in case_group.iterrows()])
        }
        
        case_stats.append(case_result)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(case_stats)
    
    print(f"Processed {len(result_df):,} unique cases")
    print(f"Sample voting percentages:")
    print(f"  Average % in favor: {result_df['pct_in_favor'].mean():.3f}")
    print(f"  Average % against: {result_df['pct_against'].mean():.3f}")
    print(f"  Average % absent: {result_df['pct_absent'].mean():.3f}")
    
    # Display some sample cases
    print("\nSample processed cases:")
    print("="*100)
    sample_cases = result_df.head(3)
    for _, case in sample_cases.iterrows():
        print(f"Case: {case['caseName']}")
        print(f"  ID: {case['caseIssuesId']}")
        print(f"  Citation: {case['usCite']}")
        print(f"  Chief: {case['chief']}")
        print(f"  Votes: {case['votes_in_favor']} in favor, {case['votes_against']} against, {case['votes_absent']} absent")
        print(f"  Percentages: {case['pct_in_favor']:.1%} in favor, {case['pct_against']:.1%} against, {case['pct_absent']:.1%} absent")
        print("-"*80)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    # Additional statistics
    print(f"\nDataset Statistics:")
    print(f"  Date range: {result_df['caseIssuesId'].str[:4].min()} - {result_df['caseIssuesId'].str[:4].max()}")
    print(f"  Unique chiefs: {result_df['chief'].nunique()}")
    print(f"  Cases with unanimous decisions (100% in favor): {(result_df['pct_in_favor'] == 1.0).sum()}")
    print(f"  Cases with no abstentions: {(result_df['pct_absent'] == 0.0).sum()}")
    
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
    
    args = parser.parse_args()
    process_cases_metadata(args.input, args.output)

if __name__ == "__main__":
    main()







