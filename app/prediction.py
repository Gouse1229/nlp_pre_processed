from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()


# Define the Pydantic model for input validation
class Email(BaseModel):
    subject: str
    content: str


# Categorization logic
def categorize_email(subject: str, content: str) -> str:
    # Combine subject and content for better context
    combined_text = f"Subject: {subject} Content: {content}"

    candidate_labels = ["Claims", "Inquiries", "Renewals", "Complaints","billing", "Other"]

    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        tokenizer="joeddav/xlm-roberta-large-xnli",
        use_fast=False
    )

    category = classifier(combined_text, candidate_labels)

    # Extract the highest-scoring category from the `category` dictionary
    highest_score_index = category['scores'].index(max(category['scores']))
    top_label = category['labels'][highest_score_index]

    return top_label
