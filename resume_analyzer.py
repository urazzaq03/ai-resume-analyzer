import os
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(file_path):
    """
    Extract text from a PDF or DOCX file.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    elif ext in ('.docx', '.doc'):
        return docx2txt.process(file_path)
    else:
        raise ValueError('Unsupported file type: {}'.format(ext))


def compute_similarity(resume_text, job_text):
    """
    Compute cosine similarity between resume and job description using TF-IDF.
    Returns a float between 0 and 1 indicating match score.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    docs = [resume_text, job_text]
    tfidf = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return similarity


def highlight_missing_keywords(resume_text, job_text, top_n=20):
    """
    Identify keywords present in the job description but missing in the resume.
    Returns a set of missing keywords (limited to top_n words).
    """
    resume_tokens = set(resume_text.lower().split())
    job_tokens = set(job_text.lower().split())
    missing = job_tokens - resume_tokens
    return set(list(missing)[:top_n])


def analyze_resume(resume_file, job_description):
    """
    Analyze a resume against a job description. Prints match score and missing keywords.

    Parameters:
        resume_file (str): Path to the resume file (PDF or DOCX).
        job_description (str): Text of the job description.
    """
    resume_text = extract_text(resume_file)
    similarity = compute_similarity(resume_text, job_description)
    missing = highlight_missing_keywords(resume_text, job_description)
    print(f"Match score: {similarity * 100:.2f}%")
    if missing:
        print("Missing keywords:", ", ".join(missing))
    else:
        print("All keywords from the job description are present in the resume!")


if __name__ == '__main__':
    # Example usage: supply a resume file path and job description text.
    # Replace 'resume.pdf' with the path to your resume file and fill job_description.
    resume_file = 'resume.pdf'
    job_description = (
        """Enter the job description here. """
    )
    analyze_resume(resume_file, job_description)
