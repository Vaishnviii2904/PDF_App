from transformers import pipeline

# Load QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_question(question, context):
    if len(context.strip()) < 50:
        return "Not enough content to answer."

    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error during QA: {str(e)}"
