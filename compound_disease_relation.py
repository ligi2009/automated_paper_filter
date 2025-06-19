import json
import os
import requests
import time
from openai import OpenAI
from tqdm import tqdm
import re

def get_pubmed_compound_cid(compound_name, max_retries=3):
    """
    Search PubChem API to get Compound CID for a given compound name
    """
    try:
        # Clean compound name for API search
        clean_name = compound_name.strip().lower()
        
        # PubChem REST API endpoint for compound search by name
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
        url = f"{base_url}/{clean_name}/cids/JSON"
        
        for attempt in range(max_retries):
            try:
                print(f"    Searching PubChem for {compound_name} (attempt {attempt+1})...")
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    if cids:
                        print(f"    Found CID: {cids[0]}")
                        return cids[0]  # Return the first CID found
                elif response.status_code == 404:
                    # Try alternative search with synonyms
                    print(f"    Trying synonym search for {compound_name}...")
                    synonym_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/synonym/{clean_name}/cids/JSON"
                    synonym_response = requests.get(synonym_url, timeout=15)
                    if synonym_response.status_code == 200:
                        synonym_data = synonym_response.json()
                        synonym_cids = synonym_data.get("IdentifierList", {}).get("CID", [])
                        if synonym_cids:
                            print(f"    Found CID via synonym: {synonym_cids[0]}")
                            return synonym_cids[0]
                    print(f"    No CID found for {compound_name}")
                    return None
                else:
                    print(f"    API returned status {response.status_code} for {compound_name}")
                
                # Add delay between retries
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
            except requests.Timeout:
                print(f"    Timeout for {compound_name} (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    time.sleep(3)
                continue
            except requests.RequestException as e:
                print(f"    Request error for {compound_name} (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                continue
        
        print(f"    Failed to get CID for {compound_name} after {max_retries} attempts")
        return None
        
    except Exception as e:
        print(f"    Error getting CID for {compound_name}: {e}")
        return None

def extract_compounds_with_gpt(abstract, client, max_retries=3):
    """
    Use GPT-4.1-mini to extract chemical compound names from abstract
    """
    if not abstract:
        return [], "No abstract available for analysis"
    
    for attempt in range(max_retries):
        try:
            print(f"    Extracting compounds with GPT (attempt {attempt+1})...")
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0.0,
                timeout=30,
                messages=[
                    {"role": "system", "content": "You are a scientific expert that identifies chemical compound names in scientific abstracts. Return your analysis in JSON format only."},
                    {"role": "user", "content": f"""Analyze the following abstract and identify any specific chemical compound names mentioned. 
                    
                    Focus on chemical substances with specific names, exclude general terms like "chemicals", "compounds", "substances"      
                    Return your response in this exact JSON format only:
                    {{
                    "compound_list": [List all specific chemical compound names found],
                    "reason": "Brief explanation of what compounds were found and why"
                    }}
                    
                    Abstract: {abstract}"""
                    }
                ],
                max_tokens=256,
                response_format={"type": "json_object"}
            )
            
            result_json = response.choices[0].message.content
            
            try:
                result = json.loads(result_json)
                compounds = result.get("compound_list", [])
                print(f"    GPT found {len(compounds)} compounds")
                return compounds, result.get("reason", "No reason provided")
            except json.JSONDecodeError as e:
                print(f"    Failed to parse JSON response: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return [], f"Error parsing response: {str(e)}"
                    
        except Exception as e:
            print(f"    Error with GPT compound extraction (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
    
    return [], f"Error during analysis after {max_retries} attempts"

def analyze_compound_disease_relation(compound, disease, abstract, client, max_retries=3):
    """
    Use GPT to analyze if a compound is related to a specific disease based on the abstract
    """
    for attempt in range(max_retries):
        try:
            print(f"    Analyzing {compound} vs {disease} (attempt {attempt+1})...")
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0.0,
                timeout=30,
                messages=[
                    {"role": "system", "content": "You are a medical and toxicology expert analyzing relationships between chemical compounds and diseases. Return your analysis in JSON format only."},
                    {"role": "user", "content": f"""Based on the following abstract, determine if there is a relationship between the compound '{compound}' and the disease '{disease}'.
                    
                    Return your response in this exact JSON format only:
                    {{
                    "relationship": "Y if there is evidence of relationship, N if no clear relationship",
                    "reason": "Detailed explanation of your decision based on the abstract content"
                    }}
                    
                    Abstract: {abstract}
                    Compound: {compound}
                    Disease: {disease}"""
                    }
                ],
                max_tokens=256,
                response_format={"type": "json_object"}
            )
            
            result_json = response.choices[0].message.content
            
            try:
                result = json.loads(result_json)
                relationship = result.get("relationship", "N")
                reason = result.get("reason", "No reason provided")
                print(f"    Result: {relationship}")
                return relationship, reason
            except json.JSONDecodeError as e:
                print(f"    Failed to parse JSON response: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "N", f"Error parsing response: {str(e)}"
                    
        except Exception as e:
            print(f"    Error with GPT relationship analysis (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
    
    return "N", f"Error during analysis after {max_retries} attempts"

def process_paper_file(file_path, client):
    """
    Process a single paper JSON file and extract compound information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        abstract = paper_data.get("Abstract", "")
        verified_list = paper_data.get("Verified_List", [])
        
        if not abstract:
            print(f"No abstract found in {file_path}")
            return None
        
        if not verified_list:
            print(f"No verified diseases found in {file_path}")
            return None
        
        # Extract compounds using GPT
        compounds, compound_reason = extract_compounds_with_gpt(abstract, client)
        
        if not compounds:
            print(f"No compounds found in {file_path}")
            # Still create output with empty compound data
            paper_data["gpt_extract_compound"] = []
            paper_data["compound_CID"] = []
            paper_data["compound_disease_relation"] = []
            paper_data["compound_extraction_reason"] = compound_reason
            return paper_data
        
        print(f"Found {len(compounds)} compounds in {os.path.basename(file_path)}: {compounds}")
        
        # Get CID for each compound
        compound_cids = {}
        for i, compound in enumerate(compounds):
            print(f"  Getting CID for compound {i+1}/{len(compounds)}: {compound}...")
            try:
                cid = get_pubmed_compound_cid(compound)
                compound_cids[compound] = cid if cid is not None else "None"
                time.sleep(1)  # Rate limiting for PubChem API
            except Exception as e:
                print(f"  Error getting CID for {compound}: {e}")
                compound_cids[compound] = "None"
        
        # Analyze compound-disease relationships
        compound_disease_relations = []
        
        for i, compound in enumerate(compounds):
            print(f"  Analyzing compound {i+1}/{len(compounds)}: {compound}")
            compound_relations = {}
            
            for j, disease in enumerate(verified_list):
                print(f"    Disease {j+1}/{len(verified_list)}: {disease}")
                try:
                    relationship, reason = analyze_compound_disease_relation(compound, disease, abstract, client)
                    
                    relation_key = f"{compound}_{disease.replace(' ', '_')}"
                    compound_relations[relation_key] = {
                        "relationship": relationship,
                        "reason": reason
                    }
                    
                    time.sleep(0.5)  # Small delay between API calls
                except Exception as e:
                    print(f"    Error analyzing {compound} vs {disease}: {e}")
                    relation_key = f"{compound}_{disease.replace(' ', '_')}"
                    compound_relations[relation_key] = {
                        "relationship": "N",
                        "reason": f"Error during analysis: {str(e)}"
                    }
            
            compound_disease_relations.append(compound_relations)
        
        # Update paper data with new information
        paper_data["gpt_extract_compound"] = compounds
        paper_data["compound_CID"] = [compound_cids]
        paper_data["compound_disease_relation"] = compound_disease_relations
        paper_data["compound_extraction_reason"] = compound_reason
        
        return paper_data
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # Configuration
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found.")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Create output directory
    os.makedirs("compound", exist_ok=True)
    
    # Get all JSON files from both directories
    input_directories = ["MRCONSO_verified", "UMLS_verified"]
    all_files = []
    
    for directory in input_directories:
        if os.path.exists(directory):
            json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
            all_files.extend(json_files)
            print(f"Found {len(json_files)} files in {directory}")
        else:
            print(f"Directory {directory} not found")
    
    if not all_files:
        print("No JSON files found in verification directories")
        return
    
    print(f"Total files to process: {len(all_files)}")
    
    # Check which files are already processed
    already_processed = set()
    if os.path.exists("compound"):
        existing_files = [f for f in os.listdir("compound") if f.endswith('.json')]
        already_processed = set(existing_files)
        print(f"Found {len(already_processed)} already processed files in compound/")
    
    # Filter out already processed files
    files_to_process = []
    skipped_count = 0
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename in already_processed:
            skipped_count += 1
        else:
            files_to_process.append(file_path)
    
    print(f"Skipping {skipped_count} already processed files")
    print(f"Will process {len(files_to_process)} new files")
    
    if len(files_to_process) == 0:
        print("All files have been processed already!")
        return
    
    # Process each file
    processed_count = 0
    error_count = 0
    
    for file_path in tqdm(files_to_process, desc="Processing papers"):
        try:
            # Get filename without extension for output
            filename = os.path.basename(file_path)
            output_path = os.path.join("compound", filename)
            
            print(f"\nProcessing {filename}...")

            result = process_paper_file(file_path, client)
            
            if result:
                # Save result
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                processed_count += 1
                print(f"✓ Saved compound analysis to {output_path}")
            else:
                error_count += 1
                print(f"✗ Failed to process {filename}")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"\n=== COMPOUND ANALYSIS SUMMARY ===")
    print(f"Total files processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved in: compound/")
    print("=====================================")

if __name__ == "__main__":
    main()