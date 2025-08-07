from transformers import pipeline
import re

# Load the model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text, max_words=600, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
    return chunks

def summarize_text(text):
    text = clean_text(text)
    chunks = split_into_chunks(text)

    all_summaries = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip().split()) < 50:
            continue  # Skip tiny chunks

        try:
            # Summarize this chunk
            result = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
            summary = result[0]['summary_text']
            all_summaries.append(f"\n{summary}")
        except Exception as e:
            all_summaries.append(f"Error summarizing chunk {i+1}: {str(e)}")

    return "\n\n".join(all_summaries)
