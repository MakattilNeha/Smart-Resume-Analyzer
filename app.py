import re
import docx2txt
from PyPDF2 import PdfReader 
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import os

DATASET_PATH = "dataset"

skills_csv1 = pd.read_csv(os.path.join(DATASET_PATH, "skills1.csv"))
skills_csv2 = pd.read_csv(os.path.join(DATASET_PATH, "all_data_skill_and_nonskills.csv"))

skills_xlsx = pd.read_excel(os.path.join(DATASET_PATH, "Technology Skills.xlsx"))

#combine all skills

all_skills = pd.concat([skills_csv1, skills_csv2, skills_xlsx], ignore_index=True)

# Convert to lowercase and make a set for fast lookup
TECHNICAL_SKILLS = set(all_skills.iloc[:, 0].str.lower().tolist())

def extract_skills(text):
    text = text.lower()
    words = set(text.split())
    skills_found = words.intersection(TECHNICAL_SKILLS)
    return skills_found

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\.\-]', "", text)
    return text

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return list(set(keywords))

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfid_matrix = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfid_matrix[0:1], tfid_matrix[1:2])[0][0]
    return round(score * 100,2)

#-------streamlit-----

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("Smart Resume Analyzer (Python + NLP)")

uploaded_resume = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc = st.text_area("Paste Job Description Here")

if uploaded_resume and job_desc:

    if uploaded_resume.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_resume)
    else:
        resume_text = extract_text_from_docx(uploaded_resume)

    jd_text = job_desc

    # clean text

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    #extract keywords

    resume_keywords = extract_skills(resume_clean)
    jd_keywords = extract_skills(jd_clean)


    #compare

    matched = set(resume_keywords).intersection(set(jd_keywords))
    missing = set(jd_keywords) - set(resume_keywords)

    #similarity store

    score = calculate_similarity(resume_clean, jd_clean)

    #display result

    st.subheader("Analysis Result")
    st.write(f" **Resume Match Score:** {score}%")
    st.write(f" **Matched Keywords:** {', '.join(matched) if matched else 'None'}")
    st.write(f" **Missing Keywords:** {', '.join(missing) if missing else 'None'}")

    #Visualization

    st.subheader("Skill Match Visualization")
    labels = ["Matched Skills", "Missing Skills"]
    values = [len(matched), len(missing)]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["green", "red"])
    ax.set_ylabel("Count")
    ax.set_title("Resume vs Job Description - Skill Match")
    st.pyplot(fig) 