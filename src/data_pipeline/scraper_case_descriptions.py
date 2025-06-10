#!/usr/bin/env python3
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re
import argparse

def citation_to_url(us_cite: str) -> str:
    """
    Convert a US citation to a Justia URL.
    Example: "329 U.S. 1" -> "https://supreme.justia.com/cases/federal/us/329/1/"
    """
    if pd.isna(us_cite) or not us_cite:
        return None
    
    # Parse citation format like "329 U.S. 1" 
    match = re.match(r'(\d+)\s+U\.S\.?\s+(\d+)', str(us_cite))
    if not match:
        print(f"  [WARNING] Could not parse citation: {us_cite}")
        return None
    
    volume = match.group(1)
    page = match.group(2)
    
    # Construct Justia URL
    url = f"https://supreme.justia.com/cases/federal/us/{volume}/{page}/"
    return url

def sanitize_filename(citation: str) -> str:
    """Convert citation to a safe filename."""
    if pd.isna(citation):
        return "unknown.txt"
    
    # Replace problematic characters
    filename = re.sub(r'[^\w\s.-]', '', str(citation))
    filename = re.sub(r'\s+', '_', filename)
    return filename + ".txt"

def scrape_case_description(url: str) -> str:
    """
    Scrape case description from Justia website.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find the main case content
        # First try to find the opinion tab content (Justia-specific)
        content_text = ""
        
        # Look for opinion tab divs (contains full Supreme Court opinion)
        opinion_tabs = soup.find_all('div', id=lambda x: x and 'tab-opinion' in x)
        if opinion_tabs:
            for tab in opinion_tabs:
                text_content = tab.get_text(strip=True)
                if len(text_content) > len(content_text):  # Get the largest one
                    content_text = text_content
        
        # Fallback: try other content selectors
        if not content_text or len(content_text) < 1000:
            content_selectors = [
                'div.opinion-content',
                'div.case-content', 
                'div.opinion',
                'div.decision-content',
                'main',
                'article',
                '.case-text',
                '.opinion-text'
            ]
            
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    temp_text = content_div.get_text(strip=True)
                    if len(temp_text) > len(content_text):  # Get the largest content
                        content_text = temp_text
        
        # Fallback: try to get text from paragraphs if no content div found
        if not content_text or len(content_text) < 100:
            paragraphs = soup.find_all('p')
            if paragraphs:
                content_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Additional fallback: get all text from body
        if not content_text or len(content_text) < 100:
            body = soup.find('body')
            if body:
                content_text = body.get_text(strip=True)
        
        # Clean up the text
        if content_text:
            # Remove excessive whitespace
            content_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', content_text)
            content_text = re.sub(r'[ \t]+', ' ', content_text)
            
            # Try to extract just the opinion/decision content
            case_start_patterns = [
                r'SUPREME COURT OF THE UNITED STATES',
                r'U\.S\. SUPREME COURT',
                r'No\. \d+',
                r'Argued.*?Decided',
                r'Syllabus',
                r'Opinion',
                r'Chief Justice.*delivered',
                r'Justice.*delivered'
            ]
            
            for pattern in case_start_patterns:
                match = re.search(pattern, content_text, re.IGNORECASE)
                if match:
                    content_text = content_text[match.start():]
                    break
        
        return content_text.strip() if content_text else ""
        
    except requests.RequestException as e:
        print(f"  [ERROR] Request failed: {e}")
        return ""
    except Exception as e:
        print(f"  [ERROR] Scraping failed: {e}")
        return ""

def process_case_descriptions(input_file: str, output_dir: str, limit: int = None):
    """
    Process cases metadata and scrape descriptions from Justia.
    """
    print(f"Loading cases metadata from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get unique citations to avoid duplicates
    unique_citations = df[['caseIssuesId', 'usCite', 'caseName']].drop_duplicates(subset=['usCite'])
    
    if limit:
        unique_citations = unique_citations.head(limit)
        print(f"Limiting to first {limit} cases for testing...")
    
    print(f"Processing {len(unique_citations):,} unique cases...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    no_content = 0
    
    for idx, row in unique_citations.iterrows():
        case_id = row['caseIssuesId']
        us_cite = row['usCite'] 
        case_name = row['caseName']
        
        print(f"[{idx + 1:,}/{len(unique_citations):,}] Processing: {case_name} ({us_cite})")
        
        try:
            # Convert citation to URL
            url = citation_to_url(us_cite)
            if not url:
                print(f"  [SKIP] Could not create URL for citation: {us_cite}")
                failed += 1
                continue
            
            print(f"  [FETCHING] {url}")
            
            # Scrape case description
            description = scrape_case_description(url)
            
            if not description:
                print(f"  [WARNING] No content retrieved")
                no_content += 1
                continue
            
            if len(description) < 100:
                print(f"  [WARNING] Content too short ({len(description)} chars)")
                no_content += 1
                continue
            
            # Create filename
            filename = sanitize_filename(us_cite)
            filepath = os.path.join(output_dir, filename)
            
            # Save description
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Case: {case_name}\n")
                f.write(f"Citation: {us_cite}\n")
                f.write(f"Case ID: {case_id}\n")
                f.write(f"Source: {url}\n")
                f.write(f"{'='*80}\n\n")
                f.write(description)
            
            successful += 1
            word_count = len(description.split())
            print(f"  [SUCCESS] Saved {filename} ({word_count:,} words)")
            
            # Rate limiting - be respectful to the website
            time.sleep(1)  # 1 second delay between requests
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {case_name}: {e}")
            failed += 1
            continue
    
    print(f"\n=== SUMMARY ===")
    print(f"Total cases processed: {len(unique_citations):,}")
    print(f"Successful: {successful:,}")
    print(f"Failed: {failed:,}")
    print(f"No content: {no_content:,}")
    print(f"Success rate: {successful/len(unique_citations)*100:.1f}%")
    print(f"Case descriptions saved to: {output_dir}")
    
    return successful

def main():
    parser = argparse.ArgumentParser(
        description="Scrape case descriptions from Justia using US citations"
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
        default="data/external/case_descriptions",
        help="Directory to save scraped case descriptions"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of cases to process (for testing)"
    )
    
    args = parser.parse_args()
    process_case_descriptions(args.input, args.output, args.limit)

if __name__ == "__main__":
    main()
