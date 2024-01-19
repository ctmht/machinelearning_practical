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


class APIServer:
    def __init__(self, model_path: str, w2v_path: str, run: bool = True):
        self.model = load_model(model_path)
        self.w2v = load_model(w2v_path)
        self.app = None

        if run:
            self.run()

    def run(self) -> None:
        self.app: FastAPI = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "Welcome to emoji prediction"}

        @self.app.post("/emoji_prediction/")
        async def predict_emoji(text: ModelInput) -> ModelOutput:
            preprocessed = [preprocess_tweet(text.text)]
            embedded = create_tweet_embeddings(self.w2v, preprocessed)
            prediction = baseline_predict(self.model, embedded)
            prediction = int(prediction[0])
            return ModelOutput(**{"emoji": prediction})


if __name__ == "__main__":
    api_server: APIServer = APIServer(model_path="../models/baseline_model.pkl",
                                      w2v_path="../models/w2v_model.pkl")
