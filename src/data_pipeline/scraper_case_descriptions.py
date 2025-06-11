#!/usr/bin/env python3
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def citation_to_url(us_cite: str) -> str:
    """
    Convert a US citation to a Justia URL.
    Example: "329 U.S. 1" -> "https://supreme.justia.com/cases/federal/us/329/1/"
    """
    if pd.isna(us_cite) or not us_cite:
        return None
    
    match = re.match(r'(\d+)\s+U\.S\.?\s+(\d+)', str(us_cite))
    if not match:
        print(f"  [WARNING] Could not parse citation: {us_cite}")
        return None
    
    volume = match.group(1)
    page = match.group(2)
    url = f"https://supreme.justia.com/cases/federal/us/{volume}/{page}/"
    return url

def sanitize_filename(citation: str) -> str:
    """Convert citation to a safe filename."""
    if pd.isna(citation):
        return "unknown.txt"
    
    filename = re.sub(r'[^\w\s.-]', '', str(citation))
    filename = re.sub(r'\s+', '_', filename)
    return filename + ".txt"

def setup_gemini_api():
    """Setup Google Gemini API client."""
    api_key = os.getenv('GEMMA_KEY')
    if not api_key:
        raise ValueError("GEMMA_KEY not found in environment variables. Please add it to your .env file.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    return model

def create_filter_prompt(case_name: str, citation: str) -> str:
    """Create a prompt for Gemini to filter out post-decision content."""
    return f"""You are an expert legal analyst. I will provide you with a Supreme Court case description that contains both pre-decision information (facts, background, legal issues) and post-decision information (court holdings, decisions, outcomes).

Your task is to extract ONLY the information that would have been available BEFORE the Supreme Court hearing and decision. This is critical for training a machine learning model to predict case outcomes without data leakage.

**INCLUDE (Pre-decision content):**
- Case facts and background
- Legal disputes and issues presented
- Lower court proceedings and decisions
- Procedural history up to Supreme Court
- Technical details (for patent cases, etc.)
- Business/contractual background
- Parties involved and their positions
- Legal framework and statutes involved
- Constitutional questions presented

**EXCLUDE (Post-decision content):**
- Supreme Court's holdings, decisions, or rulings
- Justice opinions, concurrences, or dissents
- Final outcomes or judgments
- Any analysis of what the Court "held," "decided," "ruled," or "concluded"
- Vote counts or decision margins
- Impact or consequences of the decision
- References to the Court's reasoning or interpretation

**Instructions:**
1. Extract and rewrite the pre-decision content in a clear, coherent narrative
2. Focus on factual background and legal issues, not outcomes
3. Keep technical details and procedural history
4. Remove all Supreme Court decision language
5. Maintain the essential context needed to understand the legal dispute
6. Aim for 200-400 words of clean, factual content

Case: {case_name}
Citation: {citation}

Here is the full case description to filter:

---"""

def filter_content_with_gemini(model, case_name: str, citation: str, raw_content: str) -> str:
    """Use Gemini API to filter out post-decision content."""
    try:
        prompt = create_filter_prompt(case_name, citation)
        full_prompt = f"{prompt}\n{raw_content}\n---\n\nPlease provide the filtered pre-decision content:"
        
        # Limit content length to avoid API limits
        if len(full_prompt) > 30000:  # Conservative limit
            # Truncate raw content but keep prompt
            available_space = 30000 - len(prompt) - 200
            truncated_content = raw_content[:available_space] + "... [content truncated]"
            full_prompt = f"{prompt}\n{truncated_content}\n---\n\nPlease provide the filtered pre-decision content:"
        
        response = model.generate_content(full_prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            print(f"  [WARNING] Empty response from Gemini API")
            return ""
            
    except Exception as e:
        print(f"  [ERROR] Gemini API call failed: {e}")
        return ""

def scrape_full_case_description(url: str) -> str:
    """
    Scrape the complete case description from Justia website.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Setup retry strategy
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find the main case content - get everything
        raw_content = ""
        
        # Look for opinion tab divs (contains full Supreme Court opinion)
        opinion_tabs = soup.find_all('div', id=lambda x: x and 'tab-opinion' in x)
        if opinion_tabs:
            for tab in opinion_tabs:
                text_content = tab.get_text(strip=True)
                if len(text_content) > len(raw_content):
                    raw_content = text_content
        
        # Fallback: try other content selectors
        if not raw_content or len(raw_content) < 1000:
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
                    if len(temp_text) > len(raw_content):
                        raw_content = temp_text
        
        # Fallback: try to get text from paragraphs
        if not raw_content or len(raw_content) < 100:
            paragraphs = soup.find_all('p')
            if paragraphs:
                raw_content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Final fallback: get all text from body
        if not raw_content or len(raw_content) < 100:
            body = soup.find('body')
            if body:
                raw_content = body.get_text(strip=True)
        
        # Clean up the raw text
        if raw_content:
            raw_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', raw_content)
            raw_content = re.sub(r'[ \t]+', ' ', raw_content)
        
        return raw_content.strip() if raw_content else ""
        
    except requests.RequestException as e:
        print(f"  [ERROR] Request failed: {e}")
        return ""
    except Exception as e:
        print(f"  [ERROR] Scraping failed: {e}")
        return ""

def process_case_descriptions(input_file: str, output_dir: str, limit: int = None, verbose: bool = True, quiet: bool = False):
    """
    Process cases metadata and scrape descriptions, using Gemini API for content filtering.
    """
    if verbose:
        print(f"Loading cases metadata from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Get unique citations to avoid duplicates
    unique_citations = df[['caseIssuesId', 'usCite', 'caseName']].drop_duplicates(subset=['usCite'])
    
    if limit:
        unique_citations = unique_citations.head(limit)
        if verbose:
            print(f"Limiting to first {limit} cases for testing...")
    
    if not quiet:
        print(f"Processing {len(unique_citations):,} unique cases with AI filtering...")
    
    # Setup Gemini API
    try:
        model = setup_gemini_api()
        if verbose:
            print("✅ Gemini API configured successfully")
    except Exception as e:
        print(f"❌ Failed to setup Gemini API: {e}")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    no_content = 0
    api_failed = 0
    
    for idx, row in unique_citations.iterrows():
        case_id = row['caseIssuesId']
        us_cite = row['usCite'] 
        case_name = row['caseName']
        
        if verbose:
            print(f"[{idx + 1:,}/{len(unique_citations):,}] Processing: {case_name} ({us_cite})")
        
        try:
            # Convert citation to URL
            url = citation_to_url(us_cite)
            if not url:
                if verbose:
                    print(f"  [SKIP] Could not create URL for citation: {us_cite}")
                failed += 1
                continue
            
            if verbose:
                print(f"  [FETCHING] {url}")
            
            # Scrape full case description
            raw_description = scrape_full_case_description(url)
            
            if not raw_description:
                if verbose:
                    print(f"  [WARNING] No content retrieved from website")
                no_content += 1
                continue
            
            if len(raw_description) < 500:
                if verbose:
                    print(f"  [WARNING] Content too short ({len(raw_description)} chars)")
                no_content += 1
                continue
            
            if verbose:
                print(f"  [RAW] Retrieved {len(raw_description):,} characters")
                print(f"  [GEMINI] Filtering content with AI...")
            
            # Filter content with Gemini API
            filtered_description = filter_content_with_gemini(model, case_name, us_cite, raw_description)
            
            if not filtered_description:
                if verbose:
                    print(f"  [ERROR] Gemini API filtering failed")
                api_failed += 1
                continue
            
            if len(filtered_description) < 50:
                if verbose:
                    print(f"  [WARNING] Filtered content too short ({len(filtered_description)} chars)")
                api_failed += 1
                continue
            
            # Create filename
            filename = sanitize_filename(us_cite)
            filepath = os.path.join(output_dir, filename)
            
            # Save filtered description
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Case: {case_name}\n")
                f.write(f"Citation: {us_cite}\n")
                f.write(f"Case ID: {case_id}\n")
                f.write(f"Source: {url}\n")
                f.write(f"Note: Content filtered by Gemini 2.0 Flash to exclude post-decision information\n")
                f.write(f"Raw length: {len(raw_description):,} chars, Filtered length: {len(filtered_description):,} chars\n")
                f.write(f"{'='*80}\n\n")
                f.write(filtered_description)
            
            successful += 1
            word_count = len(filtered_description.split())
            if verbose:
                print(f"  [SUCCESS] Saved {filename} ({word_count:,} words, AI-filtered)")
            
            # Rate limiting - be respectful to both APIs
            time.sleep(2)  # 2 second delay between requests (Gemini API + Justia)
            
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Failed to process {case_name}: {e}")
            failed += 1
            continue
    
    if not quiet:
        print(f"Successfully processed {successful}/{len(unique_citations)} cases ({successful/len(unique_citations)*100:.1f}% success rate)")
    
    if verbose:
        print(f"\n=== DETAILED SUMMARY ===")
        print(f"Total cases processed: {len(unique_citations):,}")
        print(f"Successful (AI-filtered): {successful:,}")
        print(f"Failed (scraping): {failed:,}")
        print(f"No content: {no_content:,}")
        print(f"API filtering failed: {api_failed:,}")
        print(f"Overall success rate: {successful/len(unique_citations)*100:.1f}%")
        print(f"Case descriptions saved to: {output_dir}")
        print(f"NOTE: All content has been filtered by Gemini 2.0 Flash AI to remove post-decision information")
    
    return successful

def main():
    parser = argparse.ArgumentParser(
        description="Scrape case descriptions from Justia and filter with Gemini 2.0 Flash API"
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
        default="data/raw/case_descriptions_ai_filtered",
        help="Directory to save AI-filtered case descriptions"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of cases to process (for testing)"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    process_case_descriptions(args.input, args.output, args.limit, verbose=verbose, quiet=args.quiet)

if __name__ == "__main__":
    main() 