#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import json
import argparse
load_dotenv()

def parse_date(s):
    """Try parsing a date like 'April 8, 1789' or return None."""
    for fmt in ("%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None

def get_justices(verbose=True):
    wiki_url = os.getenv("JUSTICES_LIST_URL", "https://en.wikipedia.org/wiki/List_of_justices_of_the_Supreme_Court_of_the_United_States")
    resp = requests.get(wiki_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # find the first sortable wikitable on the page
    table = soup.find("table", {"class": "wikitable sortable"})
    
    # If not found, try alternative approaches
    if table is None:
        if verbose:
            print("Could not find table with class 'wikitable sortable'")
        # Try finding any wikitable
        table = soup.find("table", {"class": "wikitable"})
        if table is None:
            if verbose:
                print("Could not find any wikitable")
                # List all tables for debugging
                all_tables = soup.find_all("table")
                print(f"Found {len(all_tables)} tables total")
                for i, t in enumerate(all_tables[:5]):  # Show first 5
                    classes = t.get("class", [])
                    print(f"  Table {i}: classes = {classes}")
            return []
    
    rows = table.find_all("tr")[1:]  # skip header

    justices = {}
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 7:  # Need at least 7 columns for the data we want
            continue

        try:
            # Extract name and link from column 2 (not column 1 -> has the picture)
            name_cell = cols[2]  # Changed from cols[1] to cols[2]
            a = name_cell.find("a", href=True)
            if not a:
                continue
            
            name = a.get_text(strip=True)
            # Get the justice's page URL 
            link = "https://en.wikipedia.org" + a["href"]
            
            # Debug: Print the first few URLs to check for issues
            if len(justices) < 10 and verbose:
                print(f"DEBUG: {name} -> {link}")
            
            # Extract birth/death years from the small tag in the name cell
            small_tag = name_cell.find("small")
            birth_death = small_tag.get_text(strip=True) if small_tag else ""

            # Column 3 -> State
            state_cell = cols[3]
            state_link = state_cell.find("a")
            state = state_link.get_text(strip=True) if state_link else state_cell.get_text(strip=True)

            # Column 4 -> Position (Chief Justice / Associate Justice)
            position_text = cols[4].get_text(strip=True) if len(cols) > 4 else ""
            if "Chief" in position_text:
                position = "Chief Justice"
            else:
                position = "Associate Justice"

            # Column 5 -> Who they replaced
            replaced_text = cols[5].get_text(strip=True) if len(cols) > 5 else ""

            # Column 6 -> Appointment date and method
            appointment_cell = cols[6] if len(cols) > 6 else None
            appointment_text = appointment_cell.get_text(" ", strip=True) if appointment_cell else ""
            
            # Try to extract appointment date
            appointment_date = ""
            appointment_method = ""
            if appointment_text:
                lines = appointment_text.split('\n')
                if lines:
                    appointment_date = lines[0].strip()
                    if len(lines) > 1:
                        appointment_method = lines[1].strip().strip('"()')

            # Column 7 -> Tenure period (start - end dates)
            tenure_period_text = cols[7].get_text(" ", strip=True) if len(cols) > 7 else ""
            
            # Parse tenure start and end dates
            tenure_start = ""
            tenure_end = ""
            tenure_status = ""
            
            if tenure_period_text and "–" in tenure_period_text:
                # Split on the dash to get start and end
                parts = tenure_period_text.split("–", 1)
                if len(parts) == 2:
                    tenure_start = parts[0].strip()
                    end_part = parts[1].strip()
                    
                    # Extract status (Died, Resigned, etc.) from parentheses
                    if "(" in end_part and ")" in end_part:
                        paren_start = end_part.find("(")
                        tenure_end = end_part[:paren_start].strip()
                        tenure_status = end_part[paren_start+1:end_part.find(")")].strip()
                    else:
                        tenure_end = end_part

            # Column 8 -> Tenure length
            tenure_length = cols[8].get_text(strip=True) if len(cols) > 8 else ""

            # Column 9 -> Nominating president
            nominated_by = cols[9].get_text(strip=True) if len(cols) > 9 else ""

            justices[name] = {
                "name": name,
                "url": link,
                "birth_death": birth_death,
                "state": state,
                "position": position,
                "appointment_date": appointment_date,
                "appointment_method": appointment_method,
                "replaced": replaced_text,
                "nominated_by": nominated_by,
                "tenure_start": tenure_start,
                "tenure_end": tenure_end,
                "tenure_status": tenure_status,
                "tenure_length": tenure_length
            }
            
        except Exception as e:
            if verbose:
                print(f"Error processing row: {e}")
            continue

    return justices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Supreme Court justices metadata from Wikipedia")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    parser.add_argument("--output", "-o", default="data/raw/justices.json", help="Output JSON file")
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    justices = get_justices(verbose=verbose)
    
    if not args.quiet:
        print(f"Found {len(justices)} justices")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save justices to a JSON file
    with open(args.output, "w") as f:
        json.dump(justices, f)
    
    if not args.quiet:
        print(f"Justice metadata saved to: {args.output}")
    
    # Print sample of justices only if verbose
    if verbose:
        print("=" * 80)
        print("Sample justices:")
        i = 0
        for name, justice in justices.items():
            print(f"{name}:")
            print(f"  URL:              {justice['url']}")
            print(f"  State:            {justice['state']}")
            print(f"  Position:         {justice['position']}")
            print(f"  Nominated by:     {justice['nominated_by']}")
            print(f"  Appointment Date: {justice['appointment_date']}")
            print(f"  Tenure Start:     {justice['tenure_start']}")
            print(f"  Tenure End:       {justice['tenure_end']} ({justice['tenure_status']})")
            print(f"  Tenure Length:    {justice['tenure_length']}")
            print(f"  Replaced:         {justice['replaced']}")
            print("-" * 80)
            i += 1
            if i > 10:
                break