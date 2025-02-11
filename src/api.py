import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
import os

from misinformation_modeler import MisinformationModeler, model_configs, DataConfig
import utils

# Check if we're not in a docker container
if not os.path.isfile('/.dockerenv'):
    # Load environment variables
    load_dotenv()

# Define API
app = FastAPI()

data_config = DataConfig(
    path=os.path.join(utils.get_root_dir(), os.getenv("DATA_PATH")), 
    nltk_downloads=['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'],
    column_names=['Tweet', 'Classification'],
    rename_map={'Tweet': 'text', 'Classification': 'classification'},
    pickle_path=os.path.join(utils.get_root_dir(), os.getenv("PICKLE_PATH_WIN")) if os.name == 'nt' else os.path.join(utils.get_root_dir(), os.getenv("PICKLE_PATH_LINUX")),
)

# Define and run the modeler
modeler = MisinformationModeler(data_configs=data_config, model_configs=model_configs)

@app.get('/')
def index():
    return {'message': 'Welcome to the tweet classification model'}

@app.post('/predict', description='Predict the misinformation label of a tweet')
def classify(text: str): 
    misinformation = modeler.predict(text)
    return {'misinformation': int(misinformation)}       
    
@app.post('/updateLabel', description='Update the label of a tweet')
def updateLabel(text: str, original_label: int, user_label: int):
    if original_label == user_label:
        return {
            'updated': False,
            'message': 'Label is already correct'
        }
    
    return {
        'updated': modeler.update_label(text, original_label, user_label),
        'message': 'Label updated'
    }


if __name__ == '__main__':
    # if model.pkl is not in the directory, run the modeler
    if not os.path.isfile(data_config.pickle_path):
        print('Running modeler, this might take a while...')
        modeler.run()
    uvicorn.run(app, host=os.getenv("HOSTNAME"), port=int(os.getenv("PORT")), log_level="info")
