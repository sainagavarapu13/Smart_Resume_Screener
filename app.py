# Resume Screening with AI (TF-IDF + Cosine Similarity + Streamlit UI)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx2txt


# ------------------- Helper Functions ------------------- #

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""


def extract_text_from_docx(docx_file):
    try:
        text = docx2txt.process(docx_file)
        return text.strip()

    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""


def read_resume(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    return ""


def compute_similarity(resume_texts, job_description):
    corpus = resume_texts + [job_description]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:-1]

    scores = cosine_similarity(resume_vectors, job_vector)

    return scores.flatten()


# ------------------- Streamlit UI ------------------- #

st.set_page_config(
    page_title="👁️ Vibriss",
    layout="wide"
)

st.title("👁️ Vibriss - AI-Powered Resume Screener")

st.write(
    "This ATS-enabled model intelligently scores resumes to identify the most suitable candidates."
)

st.write(
    "It automatically evaluates resumes against a Job Description using TF-IDF and Cosine Similarity."
)

# ------------------- Sidebar ------------------- #

st.sidebar.header("📄 Job Description")

job_description = st.sidebar.text_area(
    "Paste Job Description Here"
)

st.sidebar.header("📁 Candidate Resumes")

resume_files = st.sidebar.file_uploader(
    "Upload Multiple Resumes (PDF/DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ------------------- Screening ------------------- #

if st.sidebar.button("⚙️ Run Screening"):

    if not job_description.strip():
        st.warning("Please provide a Job Description.")
        st.stop()

    if not resume_files:
        st.warning("Please upload at least one resume.")
        st.stop()

    st.info("🔍 Screening resumes... Please wait.")

    resume_texts = []
    candidate_names = []

    for file in resume_files:

        try:
            file.seek(0)

            text = read_resume(file)

            if text.strip():
                resume_texts.append(text)
                candidate_names.append(file.name)

            else:
                candidate_names.append(
                    file.name + " (No Text Extracted)"
                )
                resume_texts.append("")

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if not any(text.strip() for text in resume_texts):
        st.error(
            "Could not extract text from any uploaded resume."
        )
        st.stop()

    try:
        scores = compute_similarity(
            resume_texts,
            job_description
        )

        results_df = pd.DataFrame({
            "Candidate Name": candidate_names,
            "Match Score (%)": np.round(scores * 100, 2)
        })

        results_df = results_df.sort_values(
            by="Match Score (%)",
            ascending=False
        ).reset_index(drop=True)

        st.success("✅ Screening Complete!")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        csv_data = results_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="resume_scores.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error during screening: {e}")
