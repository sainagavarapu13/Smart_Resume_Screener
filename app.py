import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PyPDF2 import PdfReader
import docx2txt

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(
    page_title="👁️ Vibriss",
    layout="wide"
)

# ---------------------------
# Skills Database
# ---------------------------

SKILLS = [
    "c", "c++", "java", "python", "sql",
    "html", "css", "javascript",
    "react", "node", "express",
    "mongodb", "mysql",
    "oop",
    "data structures",
    "algorithms",
    "dbms",
    "operating systems",
    "computer networks",
    "git",
    "machine learning",
    "deep learning",
    "pandas",
    "numpy",
    "scikit-learn"
]

# ---------------------------
# Helper Functions
# ---------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9+# ]', ' ', text)
    return text


def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""


def extract_text_from_docx(docx_file):
    try:
        return docx2txt.process(docx_file)

    except Exception as e:
        st.error(f"DOCX Error: {e}")
        return ""


def read_resume(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    return ""


def extract_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return list(set(found))


def skill_match_score(resume_text, jd_text):

    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    if len(jd_skills) == 0:
        return 0

    matched = resume_skills.intersection(jd_skills)

    return len(matched) / len(jd_skills)


def compute_similarity(resume_texts, job_description):

    corpus = resume_texts + [job_description]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vector = tfidf_matrix[-1]

    resume_vectors = tfidf_matrix[:-1]

    scores = cosine_similarity(
        resume_vectors,
        job_vector
    )

    return scores.flatten()


# ---------------------------
# UI
# ---------------------------

st.title("👁️ Vibriss - AI Resume Screener")

st.write(
    "AI-powered ATS Resume Screening using TF-IDF, Cosine Similarity and Skill Matching."
)

# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.header("📄 Job Description")

job_description = st.sidebar.text_area(
    "Paste Job Description Here",
    height=250
)

st.sidebar.header("📁 Upload Resumes")

resume_files = st.sidebar.file_uploader(
    "PDF / DOCX",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ---------------------------
# Screening
# ---------------------------

if st.sidebar.button("🚀 Run Screening"):

    if not job_description.strip():
        st.warning("Enter Job Description")
        st.stop()

    if not resume_files:
        st.warning("Upload at least one resume")
        st.stop()

    st.info("🔍 Screening resumes...")

    resume_texts = []
    candidate_names = []
    extracted_skills_list = []

    for file in resume_files:

        file.seek(0)

        text = read_resume(file)

        if text.strip():

            cleaned = clean_text(text)

            resume_texts.append(cleaned)

            candidate_names.append(file.name)

            extracted_skills_list.append(
                ", ".join(extract_skills(cleaned))
            )

        else:

            resume_texts.append("")

            candidate_names.append(
                file.name + " (No Text Found)"
            )

            extracted_skills_list.append("")

    job_description = clean_text(job_description)

    tfidf_scores = compute_similarity(
        resume_texts,
        job_description
    )

    final_scores = []

    for i in range(len(resume_texts)):

        skill_score = skill_match_score(
            resume_texts[i],
            job_description
        )

        final_score = (
            0.7 * tfidf_scores[i]
            +
            0.3 * skill_score
        )

        final_score = min(final_score * 250, 100)

        final_scores.append(round(final_score, 2))

    results_df = pd.DataFrame({

        "Candidate Name": candidate_names,

        "ATS Score (%)": final_scores,

        "Skills Found": extracted_skills_list

    })

    results_df = results_df.sort_values(
        by="ATS Score (%)",
        ascending=False
    )

    st.success("✅ Screening Complete!")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    csv = results_df.to_csv(index=False)

    st.download_button(
        "📥 Download Results CSV",
        csv,
        "resume_scores.csv",
        "text/csv"
    )
