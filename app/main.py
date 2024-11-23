from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.prediction import categorize_email

# Initialize FastAPI app
app = FastAPI(title="Email Categorization API", version="1.0")

# Input schema
class EmailInput(BaseModel):
    subject: str
    content: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Email Categorization API!"}

# Categorize email endpoint
@app.post("/categorize")
def categorize_email_endpoint(email: EmailInput):
    try:
        category = categorize_email(email.subject, email.content)
        return {"subject": email.subject, "content": email.content, "category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))