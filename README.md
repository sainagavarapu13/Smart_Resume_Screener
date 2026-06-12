🧠 AI-Powered Resume Screener
This is a highly intelligent resume screening tool that uses TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity to automatically match and rank candidate resumes against a given job description. The solution includes a user-friendly Streamlit web interface, making it easy for recruiters and HR professionals to assess resume-job fit without manual scanning.

This intelligently developed AI-powered resume screening model which I built for seamless recruitment of talents and exceptionally seasoned individuals can be accessed live on streamlit

🔧 How It Works
1. Resume Text Extraction
Resumes are parsed using:

PyPDF2 for PDFs
docx2txt for DOCX files
2. Job Description Input
The job description is pasted into a sidebar text box.

3. TF-IDF Vectorization
All resume texts and the job description are converted into TF-IDF vectors using TfidfVectorizer from scikit-learn.

4. Cosine Similarity Calculation
The similarity score between each resume and the job description is computed using cosine similarity.

5. Ranking and Display
Candidates are ranked by score and displayed in descending order. Scores are downloadable as a CSV.

📌 Example Use Case
Imagine you're hiring a Data Analyst. You paste the job description, upload 10 resumes, and instantly see which candidates best match your job post — saving hours of manual review.

