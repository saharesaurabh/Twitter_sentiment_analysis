from fastapi import FastAPI
from pydantic import BaseModel
from utils.utils import cleaning_data_, predict_sentiment
import uvicorn

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    predicted_sentiment: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Twitter Sentiment Analysis API!"}

@app.post("/predict")
def predict_sentiment_endpoint(request: TextRequest):
    text = request.text
    cleaned_text = cleaning_data_(text)
    sentiment = predict_sentiment(cleaned_text)
    return TextResponse(text=text, predicted_sentiment=sentiment)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)