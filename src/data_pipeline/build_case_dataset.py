#!/usr/bin/env python3
import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def extract_justice_name_from_vote_string(justice_vote: str) -> str:
    """
    Extract justice name from vote string like 'HHBurton:2.0'
    Returns the justice name in the format expected for bio filename.
    """
    # Split by colon to get the justice code
    justice_code = justice_vote.split(':')[0]
    
    # Dictionary to map justice codes to bio filenames
    # This mapping covers all justices that appear in the dataset
    justice_mapping = {
        # Early Court Justices
        'JJay': 'John_Jay',
        'JRutledge': 'John_Rutledge', 
        'WCushing': 'William_Cushing',
        'JWilson': 'James_Wilson',
        'JBlair': 'John_Blair',
        'JIredell': 'James_Iredell',
        'TJohnson': 'Thomas_Johnson',
        'WPaterson': 'William_Paterson',
        'SChase': 'Samuel_Chase',
        'OEllsworth': 'Oliver_Ellsworth',
        'BWashington': 'Bushrod_Washington',
        'AMoore': 'Alfred_Moore',
        'JMarshall': 'John_Marshall',
        'WJohnson': 'William_Johnson',
        'HBLivingston': 'Henry_Brockholst_Livingston',
        'TTodd': 'Thomas_Todd',
        'GDuvall': 'Gabriel_Duvall',
        'JStory': 'Joseph_Story',
        'SThompson': 'Smith_Thompson',
        'RTrimble': 'Robert_Trimble',
        'JMcLean': 'John_McLean',
        'HBaldwin': 'Henry_Baldwin',
        'JMWayne': 'James_Moore_Wayne',
        'RBTaney': 'Roger_B_Taney',
        'PPBarbour': 'Philip_P_Barbour',
        'JCatron': 'John_Catron',
        'JMcKinley': 'John_McKinley',
        'PVDaniel': 'Peter_Vivian_Daniel',
        'SNelson': 'Samuel_Nelson',
        'LWoodbury': 'Levi_Woodbury',
        'RCGrier': 'Robert_Cooper_Grier',
        'BRCurtis': 'Benjamin_Robbins_Curtis',
        'JACompbell': 'John_Archibald_Campbell',
        'NClifford': 'Nathan_Clifford',
        'NHSwayne': 'Noah_Haynes_Swayne',
        'SFMiller': 'Samuel_Freeman_Miller',
        'DDavis': 'David_Davis',
        'SJField': 'Stephen_Johnson_Field',
        'SPChase': 'Salmon_P_Chase',
        'WStrong': 'William_Strong',
        'JPBradley': 'Joseph_P_Bradley',
        'WHunt': 'Ward_Hunt',
        'MWaite': 'Morrison_Waite',
        'JMHarlan': 'John_Marshall_Harlan',
        'WBWoods': 'William_Burnham_Woods',
        'SMatthews': 'Stanley_Matthews',
        'HGray': 'Horace_Gray',
        'SBlatchford': 'Samuel_Blatchford',
        'LQCLamar': 'Lucius_QuintusCincinnatus_Lamar',
        'MFuller': 'Melville_Fuller',
        'DJBrewer': 'David_J_Brewer',
        'HBBrown': 'Henry_Billings_Brown',
        'GShiras': 'George_Shiras_Jr',
        'HEJackson': 'Howell_Edmunds_Jackson',
        'EDWhite': 'Edward_Douglass_White',
        'RPeckham': 'Rufus_W_Peckham',
        'JMcKenna': 'Joseph_McKenna',
        'OWHolmes': 'Oliver_Wendell_Holmes_Jr',
        'WRDay': 'William_R_Day',
        'WHMoody': 'William_Henry_Moody',
        'HHLurton': 'Horace_Harmon_Lurton',
        'CEHughes': 'Charles_Evans_Hughes',
        'EDWhite2': 'Edward_Douglass_White',  # When he became Chief Justice
        'WVanDevanter': 'Willis_Van_Devanter',
        'JRLamar': 'Joseph_Rucker_Lamar',
        'MPitney': 'Mahlon_Pitney',
        'JCMcReynolds': 'James_Clark_McReynolds',
        'LBrandeis': 'Louis_Brandeis',
        'JHClarke': 'John_Hessin_Clarke',
        'WHRoberts': 'William_Howard_Taft',
        'GSutherland': 'George_Sutherland',
        'PButler': 'Pierce_Butler',
        'ETSanford': 'Edward_Terry_Sanford',
        'HFStone': 'Harlan_F_Stone',
        'ORoberts': 'Owen_Roberts',
        'BNCardozo': 'Benjamin_N_Cardozo',
        
        # Modern Era Justices
        'HLBlack': 'Hugo_Black',
        'SFReed': 'Stanley_Forman_Reed',
        'FFrankfurter': 'Felix_Frankfurter',
        'WODouglas': 'William_O_Douglas',
        'FMurphy': 'Frank_Murphy',
        'JFByrnes': 'James_F_Byrnes',
        'RHJackson': 'Robert_H_Jackson',
        'WBRutledge': 'Wiley_Blount_Rutledge',
        'HHBurton': 'Harold_Hitz_Burton',
        'FMVinson': 'Fred_M_Vinson',
        'TCClark': 'Tom_C_Clark',
        'SMinton': 'Sherman_Minton',
        'EWarren': 'Earl_Warren',
        'JMHarlan2': 'John_Marshall_Harlan_II',
        'WJBrennan': 'William_J_Brennan_Jr',
        'CEWhittaker': 'Charles_Evans_Whittaker',
        'PStewart': 'Potter_Stewart',
        'BRWhite': 'Byron_White',
        'AGoldberg': 'Arthur_Goldberg',
        'AFortas': 'Abe_Fortas',
        'TMarshall': 'Thurgood_Marshall',
        'WEBurger': 'Warren_E_Burger',
        'HBlackmun': 'Harry_Blackmun',
        'LFPowell': 'Lewis_F_Powell_Jr',
        'WHRehnquist': 'William_Rehnquist',
        'JPStevens': 'John_Paul_Stevens',
        'SDOConnor': 'Sandra_Day_OConnor',
        'AScalia': 'Antonin_Scalia',
        'AMKennedy': 'Anthony_Kennedy',
        'DHSouter': 'David_Souter',
        'CThomas': 'Clarence_Thomas',
        'RBGinsburg': 'Ruth_Bader_Ginsburg',
        'SBreyer': 'Stephen_Breyer',
        'JGRoberts': 'John_Roberts',
        'SAAlito': 'Samuel_Alito',
        'SSotomayor': 'Sonia_Sotomayor',
        'EKagan': 'Elena_Kagan',
        'NMGorsuch': 'Neil_Gorsuch',
        'BMKavanaugh': 'Brett_Kavanaugh',
        'ACBarrett': 'Amy_Coney_Barrett',
        'KBJackson': 'Ketanji_Brown_Jackson',
    }
    
    return justice_mapping.get(justice_code, justice_code)

def get_case_description_path(us_cite: str, case_descriptions_dir: str) -> Optional[str]:
    """
    Find the case description file path based on US citation.
    """
    if not us_cite or us_cite == 'nan':
        return None
    
    # Convert citation like "329 U.S. 1" to filename like "329_U.S._1.txt"
    filename = us_cite.replace(' ', '_') + '.txt'
    file_path = os.path.join(case_descriptions_dir, filename)
    
    if os.path.exists(file_path):
        return file_path
    else:
        return None

def get_case_metadata_path(case_id: str, case_metadata_dir: str) -> Optional[str]:
    """
    Find the case metadata file path based on case ID.
    """
    # Case ID format should match the filename format
    filename = case_id + '.txt'
    file_path = os.path.join(case_metadata_dir, filename)
    
    if os.path.exists(file_path):
        return file_path
    else:
        return None

def get_justice_bio_path(justice_name: str, bios_dir: str) -> Optional[str]:
    """
    Find the justice biography file path.
    """
    filename = justice_name + '.txt'
    file_path = os.path.join(bios_dir, filename)
    
    if os.path.exists(file_path):
        return file_path
    else:
        return None

def parse_justice_votes(justice_votes_str: str) -> List[str]:
    """
    Parse the justice votes string and return list of justice names.
    """
    if not justice_votes_str or justice_votes_str == 'nan':
        return []
    
    justice_names = []
    # Split by semicolon and process each vote
    votes = justice_votes_str.split(';')
    for vote in votes:
        vote = vote.strip()
        if vote:
            justice_name = extract_justice_name_from_vote_string(vote)
            if justice_name not in justice_names:  # Avoid duplicates
                justice_names.append(justice_name)
    
    return justice_names

def build_dataset(csv_file: str, case_metadata_dir: str, case_descriptions_dir: str, 
                 bios_dir: str, output_file: str, verbose: bool = True, quiet: bool = False):
    """
    Build the JSON dataset from the CSV file and associated directories.
    """
    dataset = {}
    
    if verbose:
        print(f"Reading CSV file: {csv_file}")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        processed_cases = 0
        for row in reader:
            case_id = row['caseIssuesId']
            us_cite = row['usCite']
            pct_in_favor = float(row['pct_in_favor']) if row['pct_in_favor'] else 0.0
            pct_against = float(row['pct_against']) if row['pct_against'] else 0.0
            pct_absent = float(row['pct_absent']) if row['pct_absent'] else 0.0
            pct_other = float(row['pct_other']) if row['pct_other'] else 0.0
            justice_votes = row['justice_votes']
            
            # Skip if we've already processed this case ID
            if case_id in dataset:
                continue
            
            # Get file paths
            case_description_path = get_case_description_path(us_cite, case_descriptions_dir)
            case_metadata_path = get_case_metadata_path(case_id, case_metadata_dir)
            
            # Get justice names from votes
            justice_names = parse_justice_votes(justice_votes)
            
            # Get justice bio paths
            justice_bio_paths = []
            for justice_name in justice_names:
                bio_path = get_justice_bio_path(justice_name, bios_dir)
                if bio_path:
                    justice_bio_paths.append(bio_path)
            
            # Build the entry
            # Format: case_id: [justice_bio_paths, case_description_path, [%_in_favor, %_against, %_absent]]
            dataset[case_id] = [
                justice_bio_paths,  # List of paths to justice bios
                case_description_path,  # Path to case description
                [pct_in_favor, pct_against, pct_absent, pct_other]  # Convert to percentages
            ]
            
            processed_cases += 1
            if processed_cases % 1000 == 0 and verbose:
                print(f"Processed {processed_cases} cases...")
    
    if not quiet:
        print(f"Built dataset with {len(dataset)} unique cases")
    
    # Save the dataset
    if verbose:
        print(f"Saving dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    if not quiet:
        print("Dataset creation complete!")
    
    # Print some statistics
    if verbose:
        cases_with_descriptions = sum(1 for entry in dataset.values() if entry[1] is not None)
        cases_with_metadata = sum(1 for case_id, entry in dataset.items() 
                                 if get_case_metadata_path(case_id, case_metadata_dir) is not None)
        
        print(f"\nStatistics:")
        print(f"  Cases with descriptions: {cases_with_descriptions}/{len(dataset)} ({cases_with_descriptions/len(dataset)*100:.1f}%)")
        print(f"  Cases with metadata: {cases_with_metadata}/{len(dataset)} ({cases_with_metadata/len(dataset)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="Build JSON dataset mapping cases to file paths and voting percentages"
    )
    parser.add_argument(
        "--csv", 
        default="data/processed/cases_metadata.csv",
        help="Path to the cases metadata CSV file"
    )
    parser.add_argument(
        "--case-metadata-dir",
        default="data/processed/case_metadata",
        help="Directory containing case metadata files"
    )
    parser.add_argument(
        "--case-descriptions-dir",
        default="data/processed/case_descriptions", 
        help="Directory containing case description files"
    )
    parser.add_argument(
        "--bios-dir",
        default="data/processed/bios",
        help="Directory containing justice biography files"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/case_dataset.json",
        help="Output JSON file path"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    # Verify input files/directories exist
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    for dir_path, dir_name in [(args.case_metadata_dir, "case metadata"), 
                              (args.case_descriptions_dir, "case descriptions"),
                              (args.bios_dir, "bios")]:
        if not os.path.exists(dir_path):
            print(f"Error: {dir_name} directory not found: {dir_path}")
            return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    build_dataset(args.csv, args.case_metadata_dir, args.case_descriptions_dir, 
                 args.bios_dir, args.output, verbose=verbose, quiet=args.quiet)

if __name__ == "__main__":
    main() 