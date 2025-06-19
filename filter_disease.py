import requests
import pandas as pd
from tqdm import tqdm
import time
import random
import os
import json
import re
from openai import OpenAI
import argparse

def get_mesh_id_from_cui(cui, api_key):
    """
    Get MeSH ID using UMLS concept content API to find atoms with MSH source
    """
    try:
        # First approach: Use the concept's atoms endpoint to find MeSH entries
        concept_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms"
        params = {
            "sabs": "MSH",  # Only get MeSH atoms
            "apiKey": api_key
        }
        
        response = requests.get(concept_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        atoms = data.get("result", [])
        if atoms:
            # Get the first MeSH atom and extract the code
            mesh_atom = atoms[0]
            mesh_code = mesh_atom.get("code")
            if mesh_code:
                # If mesh_code is a URL, extract just the MeSH ID part
                if mesh_code.startswith("https://"):
                    # Extract the last part after the final slash
                    mesh_id = mesh_code.split("/")[-1]
                    return mesh_id
                else:
                    # If it's already just the ID, return as is
                    return mesh_code
        return "None"
        
    except Exception as e:
        print(f"Error getting MeSH ID for CUI {cui}: {e}")
        return "None"

def check_umls_exact_match(disease_name, api_key):
    base_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "string": disease_name,
        "searchType": "exact",
        "apiKey": api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("result", {}).get("results", [])
        
        # For each result, get the MeSH ID using crosswalk API
        enhanced_results = []
        for result in results:
            cui = result.get("ui")
            mesh_id = "None"
            
            if cui:
                mesh_id = get_mesh_id_from_cui(cui, api_key)
                # Add a small delay to avoid overwhelming the API
                time.sleep(0.5)
            
            # Add mesh_id to the result
            enhanced_result = result.copy()
            enhanced_result["mesh_id"] = mesh_id
            enhanced_results.append(enhanced_result)
        
        print(f"UMLS results for '{disease_name}': {len(enhanced_results)} matches found")
        return enhanced_results
    except Exception as e:
        print(f"UMLS error for '{disease_name}': {e}")
        return []

def normalize_string(text):
    """
    Normalize string for comparison by removing special characters and converting to lowercase
    """
    # Step 1: Remove the text in the ()
    normalized = re.sub(r'\s*\([^)]*\)\s*', ' ',text)
    print(f"normalized = {normalized}\n")
    # Step 2: Remove special characters but keep spaces and letters/numbers
    normalized = re.sub(r'[^\w\s]', ' ', normalized.lower())
    print(f"normalized = {normalized}\n")
    # Step 3: Replace multiple spaces with single space and strip
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    print(f"normalized = {normalized}\n")
    return normalized

def search_mrconso(disease_name, mrconso_file_path):
    """
    Search for similar strings in MRCONSO.RRF file
    Returns list of matching strings found in the file
    """
    try:
        if not os.path.exists(mrconso_file_path):
            print(f"MRCONSO.RRF file not found at: {mrconso_file_path}")
            return []
        
        normalized_disease = normalize_string(disease_name)

        found_matches = []
        
        print(f"Searching MRCONSO.RRF for: '{disease_name}' (normalized: '{normalized_disease}')")
        
        with open(mrconso_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            line_count = 0
            for line in file:
                line_count += 1
                if line_count % 10000000 == 0:  # Progress indicator for large files
                    print(f"Processed {line_count} lines...")
                
                try:
                    fields = line.strip().split('|')
                    if len(fields) > 14:  # Make sure have enough fields
                        str_field = fields[14]  # STR field is at index 14
                        
                        if str_field:
                            # Check for exact match
                            if normalized_disease == str_field.lower():
                                found_matches.append(str_field)
                                print(f"normalized_disease = {normalized_disease}\n str_field = {str_field}\n")
                                print("add 1\n")
                                continue
                            
                            # Check if normalized_disease appears as substring
                            if normalized_disease in str_field.lower():
                                found_matches.append(str_field)
                                print(f"normalized_disease = {normalized_disease}\n str_field = {str_field}\n")
                                print("add 2\n")
                                continue
                
                except Exception as e:
                    continue
        
        # Remove duplicates while preserving order
        unique_matches = []
        seen = set()
        for match in found_matches:
            if match not in seen:
                unique_matches.append(match)
                seen.add(match)
        
        print(f"Found {len(unique_matches)} matches in MRCONSO.RRF for '{disease_name}'")
        return unique_matches[:5]  # Return top 5 matches to avoid too many API calls
        
    except Exception as e:
        print(f"Error searching MRCONSO.RRF for '{disease_name}': {e}")
        return []

def verify_with_mrconso_and_umls(disease_list, mrconso_file_path, umls_api_key):
    """
    Verify diseases using MRCONSO.RRF and then UMLS API
    """
    mrconso_verified = False
    verified_list = []
    mrconso_search_results = {}
    umls_search_results = {}
    
    for disease in disease_list:
        # First search in MRCONSO.RRF
        mrconso_matches = search_mrconso(disease, mrconso_file_path)
        mrconso_search_results[disease] = mrconso_matches
        
        if mrconso_matches:
            # If we found matches in MRCONSO, verify with UMLS API
            for match in mrconso_matches:
                umls_results = check_umls_exact_match(match, umls_api_key)
                if umls_results:
                    verified_list.append(disease)
                    umls_search_results[f"{disease}_via_{match}"] = umls_results
                    mrconso_verified = True
                    break  # Found verification, no need to check other matches
                time.sleep(0.5)  # Rate limiting
        
        # Small delay between diseases
        time.sleep(0.5)
    
    return mrconso_verified, verified_list, mrconso_search_results, umls_search_results

def analyze_abstract_with_gpt(abstract, client):
    if not abstract:
        return "N", [], "No abstract available for analysis"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature = 0.0,
            messages=[
                {"role": "system", "content": "You are a helpful AI that analyzes scientific abstracts to identify specific disease names mentioned and determine if they are related to diseases. Return your analysis in JSON format only."},
                {"role": "user", "content": f"""Analyze the following abstract and identify any specific disease names mentioned. 
                
                Return your response in this exact JSON format only:
                {{
                "Y/N": "Y only if specific disease names are explicitly mentioned, otherwise N", 
                "disease_list": [List all the disease names found in the abstract],
                "reason": "Your reasoning, explaining why you classified it as Y or N and highlighting the disease mentions"
                }}
                
                If the abstract mentions general terms like "disease", "illness", or "health conditions" without naming specific diseases, return "N" for "Y/N" and an empty list for "disease_list".
                
                Abstract: {abstract}"""
                }
            ],
            max_tokens=256,
            response_format={"type": "json_object"}
        )
        
        result_json = response.choices[0].message.content
        print(result_json)
        # Parse the JSON response
        import json
        try:
            result = json.loads(result_json)
            return result.get("Y/N", "N"), result.get("disease_list", []), result.get("reason", "No reason provided")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {result_json}")
            return "N", [], f"Error parsing response: {str(e)}"
                
    except Exception as e:
        print(f"Error with GPT analysis: {e}")
        return "N", [], f"Error during analysis: {str(e)}"

def load_raw_papers(filename="raw_papers_token_based.json"):
    """Load raw paper data from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers from {filename}")
        return papers
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run fetch_papers.py first.")
        return None
    except Exception as e:
        print(f"Error loading raw papers data: {e}")
        return None

def parse_range_args():
    """Parse command line arguments for start and end paper IDs"""
    parser = argparse.ArgumentParser(description='Process papers within a specified range')
    parser.add_argument('--start', type=int, default=0, 
                        help='Starting paper ID (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='Ending paper ID (default: process all papers)')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of papers to process from start (alternative to --end)')
    
    args = parser.parse_args()

    if args.count is not None:
        args.end = args.start + args.count
    
    return args

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_range_args()
    
    # Configuration
    mrconso_file_path = "./MRCONSO.RRF"  # Update this path to MRCONSO.RRF
    
    print("\n=== PAPER FILTERING AND ANALYSIS ===")
    print(f"MRCONSO file path: {mrconso_file_path}")
    print(f"Processing range: Paper ID {args.start} to {args.end if args.end else 'end'}")
    print("====================================")
    
    # Load raw papers data
    papers = load_raw_papers()
    if papers is None:
        print("Failed to load papers. Exiting.")
        exit()
    
    # Validate and adjust range
    total_papers = len(papers)
    start_id = max(0, args.start)
    end_id = min(total_papers, args.end) if args.end is not None else total_papers
    
    if start_id >= total_papers:
        print(f"Error: Start ID {start_id} is beyond the total number of papers ({total_papers})")
        exit()
    
    if start_id >= end_id:
        print(f"Error: Start ID {start_id} must be less than end ID {end_id}")
        exit()
    
    # Slice the papers list to the specified range
    papers_to_process = papers[start_id:end_id]
    
    print(f"Total papers available: {total_papers}")
    print(f"Processing papers from ID {start_id} to {end_id-1} ({len(papers_to_process)} papers)")
    
    # Get API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    umls_api_key = os.environ.get("UMLS_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found.")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Create directories for different verification levels
    os.makedirs("UMLS_verified", exist_ok=True)
    os.makedirs("MRCONSO_verified", exist_ok=True)
    os.makedirs("unverified", exist_ok=True)

    results = []
    for idx, paper in enumerate(tqdm(papers_to_process, desc="Analyzing abstracts")):
        paper_id = start_id + idx  # Use actual paper ID based on start position
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        authors = paper.get("authors", [])
        author_names = ", ".join([author.get("name", "") for author in authors]) if authors else ""

        year = paper.get("year", "")
        url = paper.get("url", "")
        venue = paper.get("venue", "")
        
        external_ids = paper.get("externalIds", {})
        doi = external_ids.get("DOI", "")

        has_disease, disease_list, reason = analyze_abstract_with_gpt(abstract, client)
        
        # Initialize verification variables
        umls_verified = False
        mrconso_verified = False
        verified_list = []
        umls_search_results = {}
        mrconso_search_results = {}
        mrconso_umls_results = {}
        verification_method = "None"
        
        if has_disease == "Y" and disease_list:
            print(f"\nProcessing paper {paper_id}: {len(disease_list)} diseases found")
            
            # First try UMLS direct verification
            for disease in disease_list:
                umls_results = check_umls_exact_match(disease, umls_api_key)
                umls_search_results[disease] = umls_results
                
                if umls_results:  # If we found any matches
                    verified_list.append(disease)
                    umls_verified = True
            
            # If UMLS verification failed, try MRCONSO + UMLS verification
            if not umls_verified:
                print(f"UMLS direct verification failed for paper {paper_id}, trying MRCONSO verification...")
                
                mrconso_verified, mrconso_verified_list, mrconso_search_results, mrconso_umls_results = \
                    verify_with_mrconso_and_umls(disease_list, mrconso_file_path, umls_api_key)
                
                if mrconso_verified:
                    verified_list.extend(mrconso_verified_list)
                    verification_method = "MRCONSO"
            else:
                verification_method = "UMLS"
        
        paper_data = {
            "id": paper_id,
            "Title": title,
            "Abstract": abstract,
            "Authors": author_names,
            "Year": year,
            "URL": url,
            "DOI": doi,
            "Venue": venue,
            "Has_Disease": has_disease,
            "Disease_List": disease_list,
            "Reason": reason,
            "UMLS_Verified": "Y" if umls_verified else "N",
            "MRCONSO_Verified": "Y" if mrconso_verified else "N",
            "Verification_Method": verification_method,
            "Verified_List": verified_list,
        }
        results.append(paper_data)
        
        # Create JSON output with all verification information
        json_output = {
            "id": paper_id,
            "title": title,
            "Abstract": abstract,
            "Disease_List": disease_list, 
            "Reason": reason,
            "UMLS_Verified": "Y" if umls_verified else "N",
            "MRCONSO_Verified": "Y" if mrconso_verified else "N",
            "Verification_Method": verification_method,
            "Verified_List": verified_list,
            "UMLS_search_results": umls_search_results,
            "MRCONSO_search_results": mrconso_search_results,
            "MRCONSO_UMLS_results": mrconso_umls_results
        }
        
        # Save to appropriate directory based on verification status
        if umls_verified:
            # Save to UMLS_verified directory
            with open(f"UMLS_verified/{paper_id}.json", "w", encoding="utf-8") as f:
                json.dump(json_output, f, ensure_ascii=False, indent=2)
        elif mrconso_verified:
            # Save to MRCONSO_verified directory
            with open(f"MRCONSO_verified/{paper_id}.json", "w", encoding="utf-8") as f:
                json.dump(json_output, f, ensure_ascii=False, indent=2)
        elif has_disease == "Y":
            # Save to unverified directory (has diseases but couldn't verify)
            with open(f"unverified/{paper_id}.json", "w", encoding="utf-8") as f:
                json.dump(json_output, f, ensure_ascii=False, indent=2)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Create filename suffix based on processing range
    range_suffix = f"_{start_id}_{end_id-1}" if start_id > 0 or end_id < total_papers else ""
    
    # Save to CSV files with range suffix
    results_df.to_csv(f"all{range_suffix}.csv", index=False)
    print(f"Total {len(results_df)} papers have been saved to all{range_suffix}.csv")

    filtered_df = results_df[results_df["Has_Disease"] == "Y"]
    filtered_df.to_csv(f"filter{range_suffix}.csv", index=False)
    print(f"Total {len(filtered_df)} papers have been saved to filter{range_suffix}.csv")

    umls_verified_df = results_df[(results_df["Has_Disease"] == "Y") & (results_df["UMLS_Verified"] == "Y")]
    mrconso_verified_df = results_df[(results_df["Has_Disease"] == "Y") & (results_df["MRCONSO_Verified"] == "Y")]
    unverified_df = results_df[(results_df["Has_Disease"] == "Y") & 
                               (results_df["UMLS_Verified"] == "N") & 
                               (results_df["MRCONSO_Verified"] == "N")]
    
    print(f"\n=== VERIFICATION SUMMARY (Papers {start_id}-{end_id-1}) ===")
    print(f"UMLS verified papers: {len(umls_verified_df)} (saved to UMLS_verified/)")
    print(f"MRCONSO verified papers: {len(mrconso_verified_df)} (saved to MRCONSO_verified/)")
    print(f"Unverified papers with diseases: {len(unverified_df)} (saved to unverified/)")
    print(f"Total papers with diseases: {len(filtered_df)}")
    print(f"Total papers processed: {len(results_df)}")
    print(f"Processing range: {start_id} to {end_id-1}")