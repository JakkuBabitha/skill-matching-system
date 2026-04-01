# Job Application Skill Matching System

## Overview
A machine learning project that matches job applicants to job roles based on
**skill similarity** using **Cosine Similarity** and **TF-IDF Vectorization**.
The system accepts a user's skills as input and ranks candidates based on how
closely their skills match the job requirements.

## Tools & Libraries Used
- Python
- Pandas & NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- RapidFuzz (Fuzzy Matching)
- Matplotlib
- Google Colab

## Files
- `Math_final1.ipynb` — Main notebook with full implementation
- `Math_Final_CSV_cleaned.csv` — Dataset of candidates and their skills
- `Report.pdf` — Detailed project report

##  How It Works
1. User enters their skills (comma-separated)
2. Skills are **preprocessed** — cleaned, uppercased, and synonym-normalized
   (e.g. `ML` → `MACHINE LEARNING`, `AI` → `ARTIFICIAL INTELLIGENCE`)
3. **TF-IDF Vectorization** converts skills into numerical vectors
4. **Cosine Similarity** scores each candidate against the user's skills
5. Candidates above a similarity threshold of `0.25` are **accepted**
6. Top 10 matched candidates are displayed and results saved to CSV

## Visualizations
- Distribution of similarity scores (histogram)
- Top 20 most common skills in the dataset (bar chart)
- Threshold sensitivity analysis (line graph)

## Key Concepts Used
- **Cosine Similarity** — measures the angle between skill vectors
- **TF-IDF** — weighs skills by importance, not just frequency
- **Fuzzy Matching** — handles slight variations in skill names
- **Synonym Normalization** — maps abbreviations to full skill names

## How to Run
1. Upload `Math_Final_CSV_cleaned.csv` to your Colab environment
2. Run all cells in order
3. Enter your skills when prompted (e.g. `Python, Machine Learning, SQL`)
4. View matched and ranked candidates

## Output Files Generated
- `accepted_candidates_tfidf.csv` — candidates above similarity threshold
- `ranked_candidates_tfidf.csv` — all candidates ranked by similarity score
