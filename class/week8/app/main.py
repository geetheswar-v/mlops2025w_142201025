import os
import joblib
import wandb
from fastapi import FastAPI
import numpy as np
import dotenv

dotenv.load_dotenv()

WANDB_API_KEY = os.getenv('WANDB_API_KEY')
print(f'W&B API Key loaded: {WANDB_API_KEY is not None}')

app = FastAPI()
MODEL_ARTIFACT = os.environ.get('WANDER_MODEL_ARTIFACT', 'your-username/classroom-deploy/iris-rf:latest')

# Downloads the artifact and returns a loaded model
def load_model_from_wandb(artifact_ref):
    try:
        wandb.login()
    except Exception:
        pass
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    path = artifact.download()
    model_file = os.path.join(path, 'model.pkl')
    return joblib.load(model_file)

@app.on_event('startup')
def startup():
    global model
    model = load_model_from_wandb(MODEL_ARTIFACT)

@app.get('/')
def root():
    return {'status': 'ok', 'model_artifact': MODEL_ARTIFACT}

@app.post('/predict')
def predict(features: list[float]):
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)
    return {'prediction': int(pred[0])}
