from fastapi import FastAPI
from pydantic import BaseModel
from loading import load_model
from baseline import baseline_predict
from preprocessing.text_preprocessing import preprocess_tweet
from preprocessing.embeddings import create_tweet_embeddings


class ModelInput(BaseModel):
    text: str


class ModelOutput(BaseModel):
    emoji: int


app = FastAPI()
model = load_model("../models/baseline_model.pkl")
w2v = load_model("../models/w2v_model.pkl")


@app.get("/")
async def root():
    return {"message": "Welcome to emoji prediction"}


@app.post("/emoji_prediction/")
async def predict_emoji(text: ModelInput) -> ModelOutput:
    preprocessed = [preprocess_tweet(text.text)]
    embedded = create_tweet_embeddings(w2v, preprocessed)
    prediction = baseline_predict(model, embedded)
    prediction = int(prediction[0])
    output = ModelOutput(**{"emoji": prediction})
    return output
