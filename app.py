import streamlit as st
import PyPDF2
from utils import extract_text_from_pdf, extract_text_from_page
from summarizer import summarize_text
from qa import answer_question
from rag import answer_question_rag



st.set_page_config(page_title="PDF Summarizer and QA", layout="wide")
st.title("PDF Summarizer and Question Answering App")

uploaded_file = st.file_uploader("Upload a PDF File", type="pdf")

if uploaded_file is not None:
    st.success("File Successfully Uploaded")

    st.subheader("What would you like to do?")
    task = st.radio(
        "Choose an action:",
        ["Summarize Entire PDF", "Summarize a Page", "Ask a Question"]
    )

    if task == "Summarize a Page":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)

        selected_page = st.number_input(
            f"Select page (1 to {num_pages})",
            min_value=1,
            max_value=num_pages,
            step=1
        )

        if st.button("Summarize Page"):
            page_text = extract_text_from_page(uploaded_file, selected_page - 1)

            if page_text:
                with st.spinner("Summarizing..."):
                    summary = summarize_text(page_text)
                st.subheader(f"Summary of Page {selected_page}")
                st.write(summary)
            else:
                st.warning("Could not extract text from that page.")

    elif task == "Summarize Entire PDF":
        if st.button("Summarize Entire Document"):
            with st.spinner("Extracting text from the full PDF..."):
                full_text = extract_text_from_pdf(uploaded_file)

            if full_text:
                with st.spinner("Summarizing full document..."):
                    summary = summarize_text(full_text)

                st.subheader("Full PDF Summary")
                st.write(summary)
            else:
                st.warning("Could not extract text from the PDF.")
    elif task == "Ask a Question":
        question = st.text_input("Enter your question:")

        if question and st.button("Get Answer"):
            with st.spinner("Extracting full text from PDF..."):
                context = extract_text_from_pdf(uploaded_file)

            if context:
                with st.spinner("Searching and answering..."):
                    answer = answer_question_rag(question, context)
                st.subheader("Answer")
                st.write(answer)
            else:
                st.warning("Could not extract text from the PDF.")

