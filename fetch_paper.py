import requests
import pandas as pd
from tqdm import tqdm
import time
import random
import json
import os

# Define search keyword
keyword = "volatile organic compound human"

# Function to make API call with token-based pagination
def fetch_semantic_scholar_with_token(search_params, max_retries=20):
    for attempt in range(max_retries):
        try:
            # Add randomized delay to avoid predictable patterns
            delay = 1 + random.random() * 2  # 1-3 second delay
            if attempt > 0:
                print(f"Attempt {attempt+1}/{max_retries}, waiting {delay:.2f} seconds...")
            time.sleep(delay)
            
            print(f"Sending request to Semantic Scholar API...")
            
            # Use the regular search endpoint with token-based pagination
            response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", 
                                    params=search_params,
                                    headers={"User-Agent": "Research Script (academic use)"})
            
            # Print response status
            print(f"Response status code: {response.status_code}")
            
            response.raise_for_status()
            
            # Parse JSON response
            json_response = response.json()
            
            # Get total number of results if available
            total = json_response.get("total", 0)
            print(f"Total results available: {total}")
            
            # Get data from response
            data = json_response.get("data", [])
            print(f"Retrieved {len(data)} papers in this API call")
            
            # Get next token for pagination
            next_token = json_response.get("token", None)
            print(f"Next token: {next_token[:50] + '...' if next_token and len(next_token) > 50 else next_token}")
            
            # Return the data, total, and next token
            return data, total, next_token
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = (2 ** attempt) + random.random() * 5  # Exponential backoff
                print(f"Rate limited (429), waiting {wait_time:.2f} seconds before retry")
                time.sleep(wait_time)
            else:
                print(f"HTTP error: {e}")
                print(f"Response: {response.text if hasattr(response, 'text') else 'No response text'}")
                if attempt == max_retries - 1:
                    return [], 0, None
                
        except Exception as e:
            print(f"Error: {e}")
            if attempt == max_retries - 1:
                return [], 0, None
    
    print(f"Failed to retrieve data after {max_retries} attempts")
    return [], 0, None

def fetch_all_papers_with_token(keyword, max_results=10000, batch_size=100):
    all_papers = []
    current_token = None
    total_available = 0
    batch_count = 0
    
    # Create a progress bar
    pbar = tqdm(total=max_results, desc="Fetching papers (token-based)")
    
    while len(all_papers) < max_results:
        batch_count += 1
        # Calculate remaining papers needed
        remaining = max_results - len(all_papers)
        current_batch_size = min(batch_size, remaining)
        
        print(f"\n--- Batch {batch_count} ---")
        print(f"Current token: {current_token[:50] + '...' if current_token and len(current_token) > 50 else current_token}")
        
        # Prepare search parameters
        search_params = {
            "query": keyword,
            "fields": "title,abstract,authors,year,url,externalIds,venue,publicationTypes",
            "limit": current_batch_size
        }
        
        # Add token parameter if we have one (for subsequent requests)
        if current_token:
            search_params["token"] = current_token
        
        print(f"Fetching batch {batch_count}, requesting {current_batch_size} papers...")
        
        # Fetch current batch of results
        current_batch, total, next_token = fetch_semantic_scholar_with_token(search_params)
        
        # Update total available if we have it
        if total and total > total_available:
            total_available = total
            # Update progress bar total if we found out there are fewer results
            if total < max_results:
                pbar.total = min(total, max_results)
                pbar.refresh()
            
        # If no more results, break the loop
        if not current_batch:
            print(f"No more results returned for batch {batch_count}, stopping pagination.")
            break
        
        # Check if we would exceed total_available
        batch_size_actual = len(current_batch)
        
        # If adding this batch would exceed total_available, only take what we need
        if total_available > 0 and len(all_papers) + batch_size_actual > total_available:
            needed = total_available - len(all_papers)
            if needed > 0:
                current_batch = current_batch[:needed]
                batch_size_actual = len(current_batch)
                print(f"Trimming batch to {batch_size_actual} papers to not exceed total_available ({total_available})")
            else:
                print(f"Already have enough papers ({len(all_papers)}), stopping.")
                break
        
        # Add current batch to all papers
        print(f"Adding {batch_size_actual} papers to collection.")
        all_papers.extend(current_batch)
        
        # Update progress bar
        pbar.update(batch_size_actual)
        
        # Check if we've reached the total available
        if total_available > 0 and len(all_papers) >= total_available:
            print(f"Reached total_available limit ({total_available}).")
            break
        
        # Check if we have a next token for pagination
        if not next_token:
            print("No next token available, reached end of results.")
            break
            
        # Update token for next request
        current_token = next_token
        
        # Add a delay between batches to respect rate limits
        delay = 2 + random.random() * 3  # 2-5 second delay
        print(f"Waiting {delay:.2f} seconds before fetching next batch...")
        time.sleep(delay)
    
    pbar.close()
    
    print(f"\nToken-based fetch complete. Retrieved {len(all_papers)} papers in total.")
    print(f"Total available from API: {total_available}")
    print(f"Processed {batch_count} batches")
    
    # Ensure we don't exceed max_results
    return all_papers[:max_results], total_available

def save_raw_papers(papers, filename="raw_papers_token_based.json"):
    """Save raw paper data to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"Raw papers data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving raw papers data: {e}")
        return False

def create_basic_csv(papers, filename="papers_token_based.csv"):
    """Create a basic CSV with paper information for quick overview"""
    try:
        paper_data = []
        for idx, paper in enumerate(papers):
            authors = paper.get("authors", [])
            author_names = ", ".join([author.get("name", "") for author in authors]) if authors else ""
            
            external_ids = paper.get("externalIds", {})
            doi = external_ids.get("DOI", "")
            
            paper_info = {
                "id": idx,
                "Title": paper.get("title", ""),
                "Abstract": paper.get("abstract", ""),
                "Authors": author_names,
                "Year": paper.get("year", ""),
                "URL": paper.get("url", ""),
                "DOI": doi,
                "Venue": paper.get("venue", ""),
            }
            paper_data.append(paper_info)
        
        df = pd.DataFrame(paper_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Basic paper information saved to {filename}")
        return True
    except Exception as e:
        print(f"Error creating basic CSV: {e}")
        return False

def create_detailed_analysis(papers, filename="papers_token_analysis.txt"):
    """Create a detailed analysis of the fetched papers"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== SEMANTIC SCHOLAR TOKEN-BASED FETCH ANALYSIS ===\n\n")
            f.write(f"Total papers fetched: {len(papers)}\n")
            f.write(f"Search keyword: '{keyword}'\n\n")
            
            # Year distribution
            years = [paper.get("year") for paper in papers if paper.get("year")]
            year_counts = {}
            for year in years:
                year_counts[year] = year_counts.get(year, 0) + 1
            
            f.write("Year Distribution:\n")
            for year in sorted(year_counts.keys(), reverse=True)[:10]:
                f.write(f"  {year}: {year_counts[year]} papers\n")
            f.write(f"  Others: {len(papers) - sum(list(year_counts.values())[:10])} papers\n\n")
            
            # Papers with abstracts
            with_abstract = sum(1 for paper in papers if paper.get("abstract"))
            f.write(f"Papers with abstracts: {with_abstract} ({with_abstract/len(papers)*100:.1f}%)\n")
            
            # Papers with DOI
            with_doi = sum(1 for paper in papers if paper.get("externalIds", {}).get("DOI"))
            f.write(f"Papers with DOI: {with_doi} ({with_doi/len(papers)*100:.1f}%)\n")
            
            # Venue distribution (top 10)
            venues = [paper.get("venue") for paper in papers if paper.get("venue")]
            venue_counts = {}
            for venue in venues:
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
            
            f.write(f"\nTop 10 Venues:\n")
            sorted_venues = sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for venue, count in sorted_venues:
                f.write(f"  {venue}: {count} papers\n")
        
        print(f"Detailed analysis saved to {filename}")
        return True
    except Exception as e:
        print(f"Error creating analysis: {e}")
        return False

# Main execution
if __name__ == "__main__":
    # Target number of papers to fetch
    target_papers = 100000
    batch_size = 1000  # Batch size for token-based API
    
    print("\n=== SEMANTIC SCHOLAR TOKEN-BASED PAPER FETCHER ===")
    print(f"Target: Fetch {target_papers} papers using token-based pagination")
    print(f"Batch size: {batch_size} papers per request")
    print(f"Search keyword: '{keyword}'")
    print("==================================================")
    
    print(f"Searching Semantic Scholar for '{keyword}' using token-based pagination...")
    
    # Fetch all papers with token-based pagination
    try:
        papers, total_available = fetch_all_papers_with_token(keyword, max_results=target_papers, batch_size=batch_size)
        
        if not papers:
            print("No results found.")
            exit()
            
        print(f"Successfully fetched {len(papers)} papers out of {total_available} total available.")
        
        # Save raw paper data
        if save_raw_papers(papers):
            print("✓ Raw paper data saved successfully")
        else:
            print("✗ Failed to save raw paper data")
            
        # Create basic CSV for overview
        if create_basic_csv(papers):
            print("✓ Basic CSV created successfully")
        else:
            print("✗ Failed to create basic CSV")
            
        # Create detailed analysis
        if create_detailed_analysis(papers):
            print("✓ Detailed analysis created successfully")
        else:
            print("✗ Failed to create detailed analysis")
            
        print(f"\n=== SUMMARY ===")
        print(f"Total papers fetched: {len(papers)}")
        print(f"Target was: {target_papers}")
        print(f"Success rate: {len(papers)/target_papers*100:.1f}%")
        print(f"Files created:")
        print(f"  - raw_papers_token_based.json (raw data)")
        print(f"  - papers_token_based_basic.csv (structured data)")
        print(f"  - papers_token_analysis.txt (analysis)")
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error during token-based paper fetching: {e}")
        import traceback
        traceback.print_exc()
        exit()