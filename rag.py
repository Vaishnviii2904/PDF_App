from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re
import nltk
def sent_tokenize(text):
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_most_relevant_chunk(question, chunks):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    best_chunk_index = scores.argmax().item()
    return chunks[best_chunk_index]


def answer_question_rag(question, full_text):
    full_text = clean_text(full_text)
    chunks = split_into_chunks(full_text)

    if not chunks:
        return " No valid content found in PDF."

    relevant_chunk = get_most_relevant_chunk(question, chunks)
    
    try:
        result = qa_pipeline(question=question, context=relevant_chunk)
        answer = result["answer"]
        start, end = result["start"], result["end"]

        # Extract the sentence containing the answer
        sentences = sent_tokenize(relevant_chunk)
        explanation = ""
        for sent in sentences:
            if answer in sent:
                explanation = sent.strip()
                break

        return f"**Answer:** {answer}\n\nExplanation: {explanation}"
    
    except Exception as e:
        return f" Error: {str(e)}"

