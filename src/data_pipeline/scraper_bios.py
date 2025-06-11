#!/usr/bin/env python3
import os
import re
import json
import requests
from bs4 import BeautifulSoup
import argparse

# Common name corrections for Wikipedia URLs
NAME_CORRECTIONS = {
    "Howell Edmunds Jackson": "Howell_E._Jackson",
    "John Marshall Harlan II": "John_Marshall_Harlan_II", 
    "John Marshall Harlan": "John_Marshall_Harlan",
    "William O. Douglas": "William_O._Douglas",
    "Byron White": "Byron_White",
    # Add more as we discover them
}

def correct_wikipedia_url(name: str, original_url: str) -> str:
    """
    Correct common Wikipedia URL mismatches between table names and actual article titles.
    """
    if name in NAME_CORRECTIONS:
        corrected_title = NAME_CORRECTIONS[name]
        corrected_url = f"https://en.wikipedia.org/wiki/{corrected_title}"
        print(f"    [CORRECTED] {name}: {original_url} -> {corrected_url}")
        return corrected_url
    return original_url

def validate_wikipedia_url(url: str) -> bool:
    """
    Check if a Wikipedia URL actually exists by making a HEAD request.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    try:
        response = requests.head(url, headers=headers, timeout=10)
        return response.status_code == 200
    except:
        return False

def slugify(name: str) -> str:
    """Turn "John Jay" → "John_Jay", removing any illegal filename chars."""
    s = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", s)

def fetch_biography(url: str) -> str:
    """
    Download the page content using Wikipedia API instead of HTML scraping.
    This should be more reliable and get the full content.
    """
    # Extract article title from Wikipedia URL
    # URL format: https://en.wikipedia.org/wiki/Article_Title
    if "/wiki/" not in url:
        return ""
    
    article_title = url.split("/wiki/", 1)[1]
    
    # Use Wikipedia API to get the full content
    api_url = "https://en.wikipedia.org/api/rest_v1/page/html/" + article_title
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    try:
        resp = requests.get(api_url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # The API returns the content differently, look for the main content
        content = soup.find("body") or soup
        
    except Exception as e:
        print(f"    [DEBUG] API failed ({e}), falling back to direct scraping")
        # Fallback to direct scraping if API fails
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Try multiple content selectors in case the structure is different
        content = soup.find("div", class_="mw-parser-output")
        if not content:
            content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            content = soup.find("div", class_="mw-content-ltr")
        if not content:
            content = soup.find("div", class_=lambda x: x and "content" in x.lower())
        
        if not content:
            print(f"    [DEBUG] No content container found for {url}")
            return ""

    # Count elements before cleanup for debugging
    original_elements = len(content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'dl']))
    
    # Remove unwanted elements that don't contain article content
    unwanted_selectors = [
        ".navbox",           # Navigation boxes
        ".ambox",            # Article message boxes
        ".hatnote",          # Disambiguation notes
        ".thumb",            # Image thumbnails and captions
        ".gallery",          # Image galleries  
        ".reflist",          # Reference lists
        ".refbegin",         # Reference sections
        ".references",       # References
        ".citation",         # Citations
        ".cite",             # Citations
        "sup.reference",     # Reference superscripts
        ".mw-editsection",   # Edit section links
        ".printfooter",      # Print footer
        ".catlinks",         # Category links
        "table.metadata",    # Metadata tables
        ".sistersitebox",    # Sister site boxes
        ".external",         # External link icons
        "table.infobox",     # Infobox tables (but keep some content)
    ]
    
    # Remove unwanted elements
    removed_count = 0
    for selector in unwanted_selectors:
        elements = content.select(selector)
        for elem in elements:
            elem.decompose()
            removed_count += 1

    bio_parts = []
    
    # Process all elements in the content
    processed_elements = content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'dl'])
    
    for elem in processed_elements:
        if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # This is a heading - use it as a section marker
            heading_text = elem.get_text(strip=True)
            if heading_text and not heading_text.startswith('[edit]'):
                bio_parts.append(f"\n=== {heading_text} ===\n")
        
        elif elem.name == 'p':
            # Regular paragraph
            text = elem.get_text(strip=True)
            if text and len(text) > 10:  # Skip very short paragraphs (likely noise)
                bio_parts.append(text + "\n")
        
        elif elem.name in ['ul', 'ol']:
            # Lists - extract list items
            list_items = []
            for li in elem.find_all('li', recursive=False):  # Only direct children
                item_text = li.get_text(strip=True)
                if item_text:
                    list_items.append(f"• {item_text}")
            
            if list_items:
                bio_parts.append("\n".join(list_items) + "\n")
        
        elif elem.name == 'dl':
            # Definition lists
            dl_items = []
            for dt in elem.find_all('dt'):
                term = dt.get_text(strip=True)
                dd = dt.find_next_sibling('dd')
                if dd:
                    definition = dd.get_text(strip=True)
                    if term and definition:
                        dl_items.append(f"{term}: {definition}")
            
            if dl_items:
                bio_parts.append("\n".join(dl_items) + "\n")

    # Join all parts and clean up excessive whitespace
    full_text = "".join(bio_parts)
    full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)
    full_text = re.sub(r'[ \t]+', ' ', full_text)
    
    # Debug output for empty results
    final_text = full_text.strip()
    if not final_text:
        print(f"    [DEBUG] Empty result - Original elements: {original_elements}, "
              f"Processed: {len(processed_elements)}, Removed: {removed_count}")
        print(f"    [DEBUG] Bio parts collected: {len(bio_parts)}")
        
        # Fallback: try to get ANY text content
        fallback_text = content.get_text(strip=True)
        if fallback_text and len(fallback_text) > 100:
            # Only show debug in verbose mode - for now just use fallback silently
            fallback_text = re.sub(r'\s+', ' ', fallback_text)
            return fallback_text[:10000]
    
    return final_text

def main(input_json: str, output_dir: str, verbose: bool = True, quiet: bool = False):
    # load the justices dict
    with open(input_json, "r", encoding="utf-8") as f:
        justices = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    failed = 0
    total = len(justices)

    if not quiet:
        print(f"Processing {total} justice biographies...")

    for name, info in justices.items():
        url = info.get("url")
        if not url:
            if verbose:
                print(f"[SKIP] {name}: no URL")
            failed += 1
            continue

        if verbose:
            print(f"[PROCESSING] {name}...")
        
        # Try to correct known URL mismatches
        corrected_url = correct_wikipedia_url(name, url)
        
        # Validate URL exists
        if not validate_wikipedia_url(corrected_url):
            if verbose:
                print(f"[ERROR] {name}: URL not accessible - {corrected_url}")
            failed += 1
            continue
            
        try:
            bio_text = fetch_biography(corrected_url)
            if not bio_text:
                if verbose:
                    print(f"[WARN] {name}: no biography extracted from {corrected_url}")
                failed += 1
                continue

            filename = slugify(name) + ".txt"
            path = os.path.join(output_dir, filename)
            with open(path, "w", encoding="utf-8") as out:
                out.write(bio_text)
            
            # Show some stats about the extracted content
            word_count = len(bio_text.split())
            char_count = len(bio_text)
            if verbose:
                print(f"[OK]   {name} → {filename} ({word_count} words, {char_count} chars)")
            successful += 1

        except Exception as e:
            if verbose:
                print(f"[ERROR] {name}: {e}")
            failed += 1

    if not quiet:
        print(f"Successfully scraped {successful}/{total} biographies ({successful/total*100:.1f}% success rate)")
    
    if verbose:
        print(f"\n=== DETAILED SUMMARY ===")
        print(f"Total justices: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/total*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia lead bios for each Supreme Court justice."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/raw/justices.json",
        help="Path to the JSON file produced by your first scraper",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/raw/bios",
        help="Directory in which to save .txt files",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet
    main(args.input, args.output, verbose=verbose, quiet=args.quiet)
