from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()


# Load pre-trained model and pipeline
def load_model():
    try:
        # Load text classification pipeline with DistilBERT
        classifier = pipeline("text-classification", model="bert-base-uncased", return_all_scores=False)
        return classifier
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")


# Instantiate the model pipeline globally
model = load_model()


# Define the Pydantic model for input validation
class Email(BaseModel):
    subject: str
    content: str


# Define categories (modify as needed)
#categories = ["Business", "Personal", "Promotional", "Support", "Spam"]


# Categorization logic
def categorize_email(subject: str, content: str) -> str:
    # Combine subject and content for better context
    combined_text = f"Subject: {subject} Content: {content}"

    # Get predictions from the model
    predictions = model(combined_text)

    # Extract the highest confidence category
    highest_prediction = max(predictions, key=lambda x: x['score'])

    # Map the label to categories if necessary (this depends on your model's output format)
    label_to_category = {
        "LABEL_0": "Claim Submission",
        "LABEL_1": "Policy Renewal",
        "LABEL_2": "Policy Inquiry",
        "LABEL_3": "Payment Issues",
        "LABEL_4": "Coverage Adjustment",
        "LABEL_5": "Customer Support",
        "LABEL_6": "Claims Status",
        "LABEL_7": "Premium Increase",
        "LABEL_8": "Policy Cancellation",
        "LABEL_9": "Fraud Detection",
        "LABEL_10": "General Inquiry",
        "LABEL_11": "Legal/Regulatory Issues"
    }

    # If you need to map the model output to custom categories, you can use the label mapping above
    category = label_to_category.get(highest_prediction['label'], "Unknown")

    return category
