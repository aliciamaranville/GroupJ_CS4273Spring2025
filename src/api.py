import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
import os

from misinformation_modeler import MisinformationModeler, model_configs, DataConfig
import utils

# Check if the application is running inside a Docker container
if not os.path.isfile('/.dockerenv'):
    # Load environment variables from .env file if not in Docker
    load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure data settings for the model
data_config = DataConfig(
    path=os.path.join(utils.get_root_dir(), os.getenv("DATA_PATH")),  # Path to dataset
    nltk_downloads=['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'],  # Required NLP resources
    column_names=['Tweet', 'Classification'],  # Column names in the dataset
    rename_map={'Tweet': 'text', 'Classification': 'classification'},  # Mapping column names to expected format
    pickle_path=os.path.join(
        utils.get_root_dir(), 
        os.getenv("PICKLE_PATH_WIN") if os.name == 'nt' else os.getenv("PICKLE_PATH_LINUX")  # Set pickle file path based on OS
    ),
)

# Initialize the misinformation modeler with the data and model configurations
modeler = MisinformationModeler(data_configs=data_config, model_configs=model_configs)

@app.get('/')
def index():
    """Root endpoint returning a welcome message."""
    return {'message': 'Welcome to the tweet classification model'}

@app.post('/predict', description='Predict the misinformation label of a tweet')
def classify(text: str): 
    """Predicts whether a given tweet is misinformation or not.

    Args:
        text: The tweet text to classify.
        
    Returns:
        An indicator of misinformation for the given text (1 for misinformation, 0 for not).
    """
    misinformation = modeler.predict(text)
    return {'misinformation': int(misinformation)}

@app.post('/updateLabel', description='Update the label of a tweet')
def updateLabel(text: str, original_label: int, user_label: int):
    """Updates the classification label of a tweet if user deems incorrect.
    
    Args:
        text: The tweet text.
        original_label: The label assigned by the model.
        user_label: The label provided by the user.
    
    Returns:
        Updates the 'updated' and 'message' status of the model.
        The 'updated' status indicates whether an update was done, and the 'message' status
        reflects the message shown to the user regarding the update.
    """
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
    # Check if the model pickle file exists; if not, train the model before starting the API
    if not os.path.isfile(data_config.pickle_path):
        print('Running modeler, this might take a while...')
        modeler.run()  # Train the model

    # Start the FastAPI application using Uvicorn with host and port from environment variables
    uvicorn.run(app, host=os.getenv("HOSTNAME"), port=int(os.getenv("PORT")), log_level="info")