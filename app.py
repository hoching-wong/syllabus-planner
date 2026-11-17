"""Streamlit entry point. Run with: streamlit run app.py"""

import streamlit as st

st.title("Syllabus Planner")
st.write("Upload your course syllabi (PDF) to build one combined timeline.")

# accept_multiple_files=True lets the user upload one syllabus per course.
# For now we only show the file names; reading them comes in a later step.
uploaded_files = st.file_uploader(
    "Syllabus PDFs", type="pdf", accept_multiple_files=True
)

for uploaded_file in uploaded_files:
    st.write(f"Received: {uploaded_file.name}")
