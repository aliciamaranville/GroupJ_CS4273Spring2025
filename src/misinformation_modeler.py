"""A module for training and evaluating misinformation detection models.

This module provides functionality for text preprocessing, feature extraction,
model training with hyperparameter tuning, and model persistence using various
ML classifiers. Designed for extendability with custom configurations.
"""

import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import vstack
import threading
import pickle
from nltk.stem.porter import PorterStemmer
import os

from utils import remove_hashtags_links_mentions, stemmer, format_lowercase_special_chars, add_lemmatization


class DataConfig:
    """Configuration container for data processing parameters.

    Attributes:
        path (str): Directory path containing input data files.
        nltk_downloads (list): NLTK resources required for processing.
        column_names (list): Relevant columns to extract from raw data.
        rename_map (dict): Column renaming mapping.
        pickle_path (str): Output path for serialized models.
    """

    path = None
    nltk_downloads = None
    column_names = None
    rename_map = None
    pickle_path = None

    def __init__(self, path, nltk_downloads, column_names, rename_map, pickle_path):
        """Initializes DataConfig with specified parameters."""
        self.path = path
        self.nltk_downloads = nltk_downloads
        self.column_names = column_names
        self.rename_map = rename_map
        self.pickle_path = pickle_path


class ModelConfig:
    """Configuration container for machine learning models.

    Attributes:
        model_class (class): Reference to sklearn model class.
        name (str): Human-readable model identifier.
        params (dict): Hyperparameter grid for grid search.
    """

    model_class = None
    name = None
    params = None

    def __init__(self, model_class, name, params):
        """Initializes ModelConfig with specified parameters."""
        self.model_class = model_class
        self.name = name
        self.params = params


class ModelResult:
    """Container for model training results and metadata.

    Attributes:
        model (object): Trained model instance.
        name (str): Model identifier from ModelConfig.
        accuracy (float): Test set accuracy score.
        best_params (dict): Optimal parameters from grid search.
        predictions (np.ndarray): Model predictions on test set.
        params (dict): Parameters used for final training.
    """

    model = None
    name = None
    accuracy = None
    best_params = None
    predictions = None
    params = None

    def __init__(self, model, name, accuracy, best_params=None, predictions=None, params=None):
        """Initializes ModelResult with training outcomes."""
        self.model = model
        self.name = name
        self.accuracy = accuracy
        self.best_params = best_params
        self.predictions = predictions
        self.params = params


class MisinformationModeler:
    """Main class implementing the misinformation detection pipeline.

    Attributes:
        data_configs (DataConfig): Data processing configuration.
        model_configs (list[ModelConfig]): List of model configurations.
        df (pd.DataFrame): Processed training data.
        X (scipy.sparse.csr_matrix): TF-IDF features matrix.
        y (pd.Series): Target labels.
        test_size (float): Proportion of data for testing.
        model_results (dict): Baseline model results without tuning.
        model_hyperparameter_tuning_results (dict): Grid search results.
        vectorizer (TfidfVectorizer): Text vectorization component.
        stemmer (PorterStemmer): Text stemming component.
        best_model (object): Best performing model instance.
    """

    data_configs, model_configs = None, None
    df = None
    X, y = None, None
    test_size = 0.20
    model_results, model_hyperparameter_tuning_results = {}, {}
    vectorizer, lemmitizer = None, None
    stemmer = PorterStemmer()
    best_model = None

    def __init__(self, data_configs: DataConfig, model_configs: list[ModelConfig]):
        """Initializes modeler with configurations and runs preprocessing.
        
        Args:
            data_configs: Data processing parameters.
            model_configs: List of model configurations.
        """
        self.data_configs = data_configs
        self.model_configs = model_configs
        self.preprocess()
        self.lemmitize()
        self.vectorize()
        self.split()

    def run(self):
        """Executes complete pipeline including training and persistence."""
        self.preprocess()
        self.lemmitize()
        self.vectorize()
        self.split()
        self.model_hyperparameter_tuning(use_threading=False)
        self.save_best_model()
        print("Done running")

    def preprocess(self, text=None) -> pd.DataFrame:
        """Processes raw text data through cleaning and normalization.
        
        Args:
            text (str, optional): Individual text sample to process.
        
        Returns:
            pd.DataFrame: Processed dataframe when no text provided.
            str: Processed text when text argument provided.
        """
        # Download nltk data
        for download in self.data_configs.nltk_downloads:
            try:
                nltk.download(download)
            except:
                print("Failed to download nltk data: " + download)

        if text is not None:
            text = remove_hashtags_links_mentions(text)
            text = format_lowercase_special_chars(text)
            return text

        # set the max width of for display purposes
        pd.set_option('display.width', 1000)

        df = pd.DataFrame()

        # Get all the files in the directory to merge data together
        for filename in os.listdir(self.data_configs.path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(self.data_configs.path, filename)
                print(f"Reading in {file_path}")

                current_df = pd.read_excel(file_path) # self.data_configs.path?
                # current_df = df[self.data_configs.column_names]
                
                df = pd.concat([df, current_df], ignore_index=True)

        df = df[self.data_configs.column_names]

        # geting only complete cases and fixing indexes
        df.dropna(axis = 0, inplace= True)

        # fixing indexes
        df = df.reset_index(drop= True)

        # Apply the function to the 'Tweet' column of the DataFrame
        df['Tweet'] = df['Tweet'].apply(remove_hashtags_links_mentions)
        df['Tweet'] = df['Tweet'].apply(format_lowercase_special_chars)

        self.df = df

        return df

    def lemmitize(self) -> pd.DataFrame:
        """Applies lemmatization to text data.
        
        Returns:
            pd.DataFrame: Dataframe with lemmatized text column.
        """
        self.df = self.df.rename(columns = self.data_configs.rename_map)
        self.df = add_lemmatization(self.df)
        return self.df

    def stem(self) -> pd.DataFrame:
        """Applies Porter stemming to text data.
        
        Returns:
            pd.DataFrame: Dataframe with stemmed text column.
        """
        self.df['text'] = self.df['text'].apply(stemmer)
        return self.df

    def vectorize(self):
        """Transforms text data to TF-IDF features matrix."""
        self.vectorizer = TfidfVectorizer()

        self.X = self.df['text']
        self.y = self.df['classification']

        self.X = self.vectorizer.fit_transform(self.X)

        return self.X, self.y

    def split(self) -> tuple:
        """Splits data into stratified train/test sets.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) split.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=2)
        # Use np.unique() to get the unique values and their counts
        unique_vals, counts = np.unique(y_train, return_counts=True)

        # Print the unique values and their counts
        for val, count in zip(unique_vals, counts):
            print("Printing the counts for each unique value in the training set.")
            print(f"{val}: {count}")

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        return X_train, X_test, y_train, y_test
    
    def train(self):
        """Trains models without hyperparameter tuning."""
        for model_config in self.model_configs:
            model = model_config.model_class(**model_config.params)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            score = accuracy_score(self.y_test, y_pred)
            print(f'Accuracy for {model_config.name}: {round(score*100,2)}%')

            self.model_results[model_config.name] = ModelResult(model, model_config.name, score)

    def model_hyperparameter_tuning(self, use_threading=True):
        """Performs grid search optimization for models.
        
        Args:
            use_threading (bool): Enable parallel execution of grid searches.
        """
        for model_config in self.model_configs:
            function_args = [model_config.model_class, model_config.name, model_config.params, self.X_train, self.X_test, self.y_train, self.y_test]
            if use_threading:
                thread = threading.Thread(target=self.run_hyperparameter_tuning, args=function_args)
                thread.start()
            else:
                self.run_hyperparameter_tuning(*function_args)

    def run_hyperparameter_tuning(self, model, model_name, params, X_train, X_test, y_train, y_test) -> None:
        """Executes grid search and stores best model.
        
        Args:
            model: Model class reference.
            model_name: Name for reporting.
            params: Parameter grid.
            X_train: Training features.
            X_test: Test features.
            y_train: Training labels.
            y_test: Test labels.
        """
        grid_search = GridSearchCV(model(), params, cv=5)

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f'Best parameters for {model_name}: ', best_params)
        print(f'Best score for {model_name}: ', best_score)

        # Training new models based on the best performing hyperparameter & testing their prediction with the testing set
        best_model = model(**best_params)

        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_true=y_test, y_pred=predictions)

        print(f'Accuracy of the testing data using {model_name}: ', accuracy)

        self.model_hyperparameter_tuning_results[model_name] = ModelResult(
            model=best_model,
            name=model_name,
            accuracy=accuracy,
            params=best_params,
        )

    def save_best_model(self):
        """Persists best performing model to disk.
        
        Raises:
            Exception: If no trained models are available.
        """
        if self.best_model is None:
            if len(self.model_hyperparameter_tuning_results) == 0 or self.model_hyperparameter_tuning_results is None:
                raise Exception('No model has been trained yet. Run the training first.')
            # Get the model with the highest accuracy
            self.best_model = max(self.model_hyperparameter_tuning_results.values(), key=lambda x: x.accuracy)
            self.best_model = self.best_model.model
        # Save the model to disk in a pickle file
        dict_to_save = {
            'model': self.best_model,
            'vectorizer': self.vectorizer,
            'stemmer': self.stemmer
        }
        with open(self.data_configs.pickle_path, 'wb') as f:
            pickle.dump(dict_to_save, f)
    
    def load_best_model(self):
        """Loads persisted model from disk.
        
        Raises:
            Exception: If model file not found.
        """
        if not os.path.exists(self.data_configs.pickle_path):
            raise Exception('Model file does not exist. Run the training first.')
        with open(self.data_configs.pickle_path, 'rb') as f:
            model_dict = pickle.load(f, fix_imports=True, encoding='latin1')
            self.best_model = model_dict['model']
            self.vectorizer = model_dict['vectorizer']
            self.stemmer = model_dict['stemmer']

    def predict(self, text: str) -> str:
        """Generates prediction for input text.
        
        Args:
            text: Raw input text to classify.
            
        Returns:
            str: Predicted class label.
        """
        if not self.best_model:
            self.load_best_model()
        text = self.vectorizer.transform([self.prepare_text_for_model(text)])
        return self.best_model.predict(text)[0]
    
    def prepare_text_for_model(self, text: str) -> str:
        """Applies full preprocessing pipeline to raw text.
        
        Args:
            text: Input text to process.
            
        Returns:
            str: Processed text ready for vectorization.
        """
        text = remove_hashtags_links_mentions(text)
        text = format_lowercase_special_chars(text)
        text = self.stemmer.stem(text)
        return text

    def update_label(self, text: str, original_label: int, user_label: int) -> bool:
        """Updates dataset with corrected labels and retrains model.
        
        Args:
            text: Text associated with label correction.
            original_label: Initial predicted label.
            user_label: Corrected label from user.
            
        Returns:
            bool: True if update occurred, False otherwise.
        """
        if original_label == user_label:
            return False
        
        text = self.preprocess(text)
        lemma = add_lemmatization(text)
        # Check if the text is already in the dataset
        existing_text_indices = np.where(self.df['text'] == text)[0]

        if len(existing_text_indices) > 0:
            for index in existing_text_indices:
                self.df['classification'][index] = user_label
        else:
            self.df = pd.concat([self.df, pd.DataFrame([[text, user_label, lemma]], columns=['text', 'classification', 'lemma_txt'])], ignore_index=True)

        self.vectorize()

        if not self.best_model:
            self.load_best_model()

        self.best_model.fit(self.X, self.y)
        # Save the new model
        self.save_best_model()
        return True


# Model configurations for grid search
model_configs = [
    ModelConfig(
        name='LogisticRegression',
        model_class=LogisticRegression,
        params={
            'C': [0.1, 0.5, 1.0, 10.0],
            'penalty': ['l1', 'l2', 'elasticnet']
        }
    ),
    ModelConfig(
        name='MultinomialNB',
        model_class=MultinomialNB,
        params={
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'fit_prior': [True, False],
        },
    ),
    ModelConfig(
        name='PassiveAggressiveClassifier',
        model_class=PassiveAggressiveClassifier,
        params={
            'C': [0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
        }
    ),
    ModelConfig(
        name='BernoulliNB',
        model_class=BernoulliNB,
        params={
            'alpha': [0.1, 0.5, 1.0, 2.0],
            'binarize': [0.0, 0.5, 1.0],
            'fit_prior': [True, False]
        }
    ),
    ModelConfig(
        name='LinearSVC',
        model_class=LinearSVC,
        params={
            'C': [0.1, 0.5, 1.0, 2.0],
            'loss': ['hinge', 'squared_hinge'],
            'penalty': ['l1', 'l2'],
            'max_iter': [500, 1000, 2000]
        }
    ),
]


# Data configuration setup
data_config = DataConfig(
    path=os.getenv("DATA_PATH"), 
    nltk_downloads=['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'],
    column_names=['Tweet', 'Classification'],
    rename_map={'Tweet': 'text', 'Classification': 'classification'},
    pickle_path=os.getenv("PICKLE_PATH_WIN") if os.name == 'nt' else os.getenv("PICKLE_PATH_LINUX")
)


if __name__ == "__main__":
    """Main execution block for running the modeling pipeline."""
    modeler = MisinformationModeler(data_config, model_configs)
    modeler.run()