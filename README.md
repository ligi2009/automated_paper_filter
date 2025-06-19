# Automated Pipeline to Fetch and Filter Papers
A pipeline that fetches papers via the Semantic Scholar API and filters their content using an LLM.

## Environment Setup
Create a new conda environment
```bash
conda create --name paper_env python=3.10
```
Activate the conda environment
```bash
conda activate paper_env
```
Install necessary package
```bash
pip insall -r requirements.txt
```

## Set API KEY
```bash
export OPENAI_API_KEY="Your openai api key"
export UMLS_API_KEY="Your umls api key"
```

## Pipeline
1. Use Semantic Scholar API to fetch papers (keyword = "volatile organic compound human")
```python fetch_paper.py```
2. Filter papers with diseases
```python filter_dusease.py```
3. Extract compound name and CID, then check the relation between compounds and diseases
```python compound_disease_relation.py```