import PyPDF2
import re

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text.strip()

def extract_text_from_page(pdf_file, page_number):
    reader = PyPDF2.PdfReader(pdf_file)
    if 0 <= page_number < len(reader.pages):
        text = reader.pages[page_number].extract_text()
        if text:
            text = re.sub(r'\s+', ' ', text)  
            text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text) 
            return text.strip()
    return ""
