# AI Resume Analyzer

An NLP-powered tool that compares a resume against a job description and returns a match score plus a list of missing keywords — helping job seekers optimize their resume for specific roles and applicant tracking systems (ATS).

## How It Works

The tool extracts text from a resume file (PDF or DOCX), then uses TF-IDF vectorization to convert both the resume and job description into weighted term vectors. Cosine similarity is computed between the two vectors to produce a match score from 0–100%. It also diffs the token sets to surface keywords present in the job description but absent from the resume.

## Features

- Supports PDF and DOCX resume formats
- TF-IDF vectorization with English stopword filtering
- Cosine similarity scoring (0–100% match)
- Missing keyword detection to target ATS gaps
- Clean modular design — each function is independently usable

## Tech Stack

- Python 3
- `scikit-learn` — TF-IDF vectorization and cosine similarity
- `PyPDF2` — PDF text extraction
- `docx2txt` — DOCX text extraction

## Usage

```python
from analyzer import analyze_resume

analyze_resume("resume.pdf", "We are looking for a Python developer with experience in NLP and machine learning...")
# Match score: 74.32%
# Missing keywords: agile, pytorch, deployment, docker
```

## Project Structure
