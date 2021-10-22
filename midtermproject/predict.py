import uvicorn
import joblib

from fastapi import FastAPI
from clean_text import clean_text_dataset
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

class NepaliText(BaseModel):
    params: str

class NepaliTextClassification:

    def __init__(self):
        self.clf = joblib.load("model.bin")
        self.tfidf = joblib.load("tfidf.bin")


    def predict(self, item: NepaliText):
        text = self.tfidf.transform([item.params])
        label = self.clf.predict(text)[0]
        return label



@app.post("/predict/")
def model_prediction(text: NepaliText):
    classification = NepaliTextClassification()
    classify = classification.predict(text)
    return {"label": classify}



if __name__ == "__main__":
    uvicorn.run("predict:app",debug = True, host = 'localhost', port = 5000, reload=True)

