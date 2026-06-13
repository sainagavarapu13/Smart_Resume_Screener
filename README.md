# 🧠 Vibriss – AI-Powered Resume Screener

Vibriss is an intelligent Applicant Tracking System (ATS) inspired resume screening platform that automatically evaluates and ranks candidate resumes against a job description using Natural Language Processing (NLP) techniques.

The system leverages TF-IDF Vectorization and Cosine Similarity to identify the most relevant candidates, helping recruiters significantly reduce manual screening effort and accelerate hiring decisions.

## 🚀 Live Demo
## 🚀 Live Demo

[Try Vibriss Resume Screener](https://smartresumescreener-o3eihhtebktmaxf9r22yjf.streamlit.app/)

## ✨ Features

* Upload multiple resumes in PDF and DOCX formats
* Automatic resume text extraction
* Job description based candidate screening
* TF-IDF based feature representation
* Cosine Similarity based resume ranking
* ATS-style candidate scoring
* Interactive Streamlit web interface
* Download screening results as CSV
* Supports bulk resume evaluation

## 🏗️ System Architecture

### 1. Resume Parsing

Candidate resumes are processed using:

* PyPDF2 for PDF extraction
* docx2txt for DOCX extraction

### 2. Job Description Input

Recruiters provide a job description through the Streamlit interface.

### 3. Text Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) transforms resumes and job descriptions into numerical feature vectors.

### 4. Similarity Analysis

Cosine Similarity measures semantic relevance between each resume and the target job description.

### 5. Candidate Ranking

Resumes are ranked according to their similarity scores and displayed in descending order.

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### Machine Learning / NLP

* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

### Data Processing

* Pandas
* NumPy

### Document Parsing

* PyPDF2
* docx2txt

## 📊 Example Workflow

1. Paste a Job Description.
2. Upload one or more resumes.
3. Run Screening.
4. View ATS Match Scores.
5. Download results as CSV.

## 📈 Future Enhancements

* Skill extraction and skill-gap analysis
* Semantic matching using Sentence Transformers
* Resume keyword highlighting
* Candidate recommendation engine
* Interactive analytics dashboard
* LLM-powered resume insights

## 🎯 Use Cases

* Campus Recruitment
* Internship Hiring
* Talent Acquisition
* Resume Shortlisting
* HR Automation
* Recruitment Analytics

## 👨‍💻 Developer

Sai Sravya Nagavarapu

* GitHub: https://github.com/sainagavarapu13
* LinkedIn: Add your LinkedIn URL here

If you found this project useful, consider giving it a ⭐ on GitHub.
