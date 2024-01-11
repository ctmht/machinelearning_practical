from fastapi import FastAPI
from pydantic import BaseModel
from loading import load_model
from baseline import baseline_predict
from preprocessing.text_preprocessing import preprocess_tweet
from preprocessing.embeddings import create_tweet_embeddings


class ModelInput(BaseModel):
    text: str


app = FastAPI()
model = load_model('../models/baseline_model.pkl')
w2v = load_model('../models/w2v_model.pkl')


@app.get('/')
async def root():
    return {'message': 'Welcome to emoji prediction'}


@app.post('/emoji_prediction/')
async def predict_emoji(text: ModelInput):
    preprocessed = [preprocess_tweet(text.text)]
    embedded = create_tweet_embeddings(w2v, preprocessed)
    prediction = baseline_predict(model, embedded)
    prediction = int(prediction[0])
    return {'emoji': prediction}

# requests can be made in the terminal with the following command:
# curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"Today is a beautiful day\"}" http://127.0.0.1:8000/emoji_prediction/
