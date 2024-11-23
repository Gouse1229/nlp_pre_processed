from fastapi import FastAPI
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Categorization logic
def categorize_email(content: str) -> str:
    # Combine subject and content for better context

    candidate_labels = ["Claims", "Inquiries", "Renewals", "Complaints","billing", "Other"]

    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        tokenizer="joeddav/xlm-roberta-large-xnli",
        use_fast=False
    )

    category = classifier(content, candidate_labels)

    # Extract the highest-scoring category from the `category` dictionary
    highest_score_index = category['scores'].index(max(category['scores']))
    top_label = category['labels'][highest_score_index]

    return top_label
