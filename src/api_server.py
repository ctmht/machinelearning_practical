from fastapi import FastAPI
from pydantic import BaseModel
from saving_loading import load
from preprocessing.text_preprocessing import preprocess_tweet
from preprocessing.embeddings import *
import numpy as np


class ModelInput(BaseModel):
    text: str


class ModelOutput(BaseModel):
    emoji: int


model = load("../models/lstm_model.pkl")
w2v = load("../models/w2v_model.pkl")
w2i_map = load("../models/w2i_map.pkl")
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to emoji prediction"}


@app.post("/emoji_prediction/")
async def predict_emoji(text: ModelInput) -> ModelOutput:
    preprocessed = [preprocess_tweet(text.text)]
    embedded = create_padded_tweet_embeddings(preprocessed, w2i_map, model.max_sequence_length)
    prediction = model.predict(embedded)[0]
    prediction = np.argmax(prediction)
    return ModelOutput(**{"emoji": prediction})
