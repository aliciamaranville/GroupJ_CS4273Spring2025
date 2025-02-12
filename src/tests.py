# Code to test functions in GitHub repo

import unittest
import pandas as pd
from sklearn.svm import LinearSVC

from misinformation_modeler import MisinformationModeler, DataConfig, ModelResult
from utils import remove_hashtags_links_mentions, format_lowercase_special_chars, create_stopwords, create_tokens_with_pos, lemmatize_word, lemmatize_tweets, add_lemmatization, stemmer
from api import index, classify, updateLabel


class TestMisinformationModeler(unittest.TestCase):
    """Tests the functions of the misinformation_modeler.py class"""

    def setUp(self):
        """Set up the testing suite using an object of the MisinformationModeler class"""

        data_config = DataConfig(
            path="data/",
            nltk_downloads=['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'],
            column_names=['Tweet', 'Classification'],
            rename_map={'Tweet': 'text', 'Classification': 'classification'},
            pickle_path="pickles/model_win.pkl"
        )
        self.model = MisinformationModeler(data_configs=data_config, model_configs=[])

    def test_preprocess(self):
        """Tests if the Modeler removes the hashtags and mentions(@someusername) from the tweets before categorizing them."""

        text = "This is a #test tweet. Another tweet with @mentions."
        expected_result = "this is a  tweet another tweet with "
        result_text = self.model.preprocess(text)
        self.assertEqual(result_text, expected_result, 'Text not preprocessed correctly.')

    def test_lemmatize(self):
        """Tests if the Modeler can add the lemma_txt column and lemmatize the words in the tweet.
        Lemmatizing the words breaks them down to their root word base."""

        result_df = self.model.lemmitize()
        expected_column_names = ['text', 'classification', 'lemma_txt']  
        self.assertEqual(list(result_df.columns), expected_column_names, 'DataFrame columns not renamed correctly.')

    def test_stem(self):
        """Tests if the Modeler can stem the words in the tweet. 
        Stemmming the words breaks off the common letters at the ends of words. 
        Similar to lemmatization, but is faster and a bit less accurate usually."""

        result_df = self.model.stem()
        self.assertIsNotNone(result_df, 'Dataframe not stemmed properly.')

    def test_vectorize(self):
        """Tests if the Modeler is able to covert raw words into TF-IFD features(numerical features
        rather than just ASCII characters) for machine learning purposes."""

        self.assertIsNotNone(self.model.vectorize(), self.model.X)
        self.assertIsNotNone(self.model.vectorize(), self.model.y)

    def test_split(self):
        """Tests if the Modeler is able to split the data into training and testing sets."""

        self.model.test_size = 0.2
        X_train, X_test, y_train, y_test = self.model.split()
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])

    def test_hyperparameter_tuning(self):
        """Tests if the Modeler is able to start threading and pass hyperparameters
        of the machine learning models to run_hyperparameter_tuning."""

        use_threading = True  
        self.model.model_hyperparameter_tuning(use_threading=use_threading)
        self.assertTrue(self.model.run_hyperparameter_tuning, 'Model not tuned correctly.')

    def test_run_hyperparameter_tuning(self):
        """Tests if the Modeler is able to tune the hyperparameters of an example LinearSVC model."""

        model = LinearSVC
        model_name = 'LinearSVC'
        params={
            'C': [0.1, 0.5, 1.0, 2.0],
            'loss': ['hinge', 'squared_hinge'],
            'penalty': ['l1', 'l2'],
            'max_iter': [500, 1000, 2000]
        }
        X_train, X_test, y_train, y_test = self.model.split() 
        self.model.run_hyperparameter_tuning(model, model_name, params, X_train, X_test, y_train, y_test)
        self.assertIsNotNone(self.model.model_hyperparameter_tuning_results[model_name])

    def test_save_model(self):
        """Tests if the Modeler can save the highest accuracy performing model
        out of the suite of models tested."""

        self.model.save_best_model()
        self.assertIsNotNone(self.model.best_model)

    def test_load_model(self):
        """Tests if the Modeler can load in a saved model from a pickle file."""

        self.model.load_best_model()
        self.assertIsNotNone(self.model.best_model)

    def test_predict(self):
        """Tests if the Modeler can predict a misinformation classification for sample text."""

        sample_text = "This is a sample text for prediction."
        prediction = self.model.predict(sample_text)
        self.assertIsNotNone(prediction, 'Model not predicting correctly.')

    def test_prepare_text(self):
        """Tests if the Modeler can remove hashtags and other unnecessary parts of tweets,
        plus stemming each word using the corresponding functions created in misinformation_modeler.py"""

        sample_text = "This is a sample text."
        prepared_text = self.model.prepare_text_for_model(sample_text)
        self.assertNotEqual(sample_text, prepared_text, 'Text not prepared correctly.')

    def test_update_label(self):
        """Tests if the Modeler can update a stored label when a user inputs a new classification
        for whether text is misinformation or not."""

        text = "This is not a test."
        original_label = 0
        user_label = 1
        self.assertTrue(self.model.update_label(text, original_label, user_label), 'Label not updated correctly.')

# End of misinformation_modeler.py functions

class TestUtils(unittest.TestCase):
    """Tests the functions of the utils.py class."""
    
    def test_remove_hastags(self):
        """Tests whether hashtags are removed from tweets."""

        self.assertEqual(remove_hashtags_links_mentions("#ou This is the biggest event."), " This is the biggest event.", 'Hastags not removed from text.')

    def test_format_lowercase(self):
        """Tests whether tweets are turned into all lowercase letters."""

        self.assertEqual(format_lowercase_special_chars("& And this is the Alamo."), " and this is the alamo", 'Text not formatted correctly.')

    def test_create_stopwords(self):
        """Tests whether the stopwords are the combination of English, Italian, and 
        Georgian stopwords."""

        self.assertNotIn('not', create_stopwords())
        common_stopwords = ['the', 'in', 'does', 'e']
        for word in common_stopwords:
            self.assertIn(word, create_stopwords())

    def test_create_tokens(self):
        """Tests whether a tweet is turned into tokens and tagged with pos."""

        self.assertEqual(create_tokens_with_pos("this is the end", create_stopwords()), [[('end', 'NN')]], 'Tokens not created correctly.')

    def test_lemmatize(self):
        """Tests whether words are being correctly lemmatized."""

        self.assertEqual(lemmatize_word('tried', 'VERB'), "try", 'Words not lemmatized correctly.') 

    def test_lemmatize_tweets(self):
        """Tests whether a list of tokens from a tweets is correctly lemmatized."""

        twt_tkn_pos_list = [[("cat", "NOUN"), ("run", "VERB")], [("dog", "NOUN"), ("jump", "VERB")]]
        expected_result = ["['cat', 'run']", "['dog', 'jump']"]
        result = lemmatize_tweets(twt_tkn_pos_list)
        self.assertEqual(result, expected_result, 'Tweets not lemmatized correctly.')

    def test_add_lemmatization(self):
        """Tests whether add_lemmatization will correctly call lemmatize_tweets when a String
        is passed as an argument."""
        
        self.assertIsInstance(add_lemmatization("This is a test sentence."), list, 'Dataframe not lemmatized correctly.')

    def test_stemmer(self):
        """Tests whether an example sentence will be stemmed correctly, even
        if the sentence uses two different languages."""

        input_text = "This is a test sentence in English and Italian. Mangiamo pizza!"
        expected_output = "test sentenc english italian mangiamo pizza"
        result = stemmer(input_text)
        self.assertEqual(result, expected_output, 'Stemming did not produce the expected output.')

# End of utils.py functions

class TestAPI(unittest.TestCase):
    """Tests the api.py class."""

    def test_index(self):
        """Tests if the home directory displays the welcome message."""
        
        result = index()
        expected_result = {'message': 'Welcome to the tweet classification model'}
        self.assertEqual(result, expected_result, 'Function did not return the expected dictionary.')

    def test_classify(self):
        """Tests if some example misinformation will be correctly classified as misinformation."""

        misinformation = classify('  reagan started a great deal of the damage cutting business taxes deregulations taxing social security etcthe only difference twixt him and tfg is that reagan was more polite about it thats it')
        self.assertEqual(misinformation, {'misinformation': 1}, 'API not classifying correctly.')

    def test_update_label(self):
        """Tests if the user requesting to update some labeled misinformation as not misinformation works."""
        
        update = updateLabel('  reagan started a great deal of the damage cutting business taxes deregulations taxing social security etcthe only difference twixt him and tfg is that reagan was more polite about it thats it', 0, 1)
        self.assertNotEqual(update, 0, 'API not updating labels correctly.')

# End of api.py functions


if __name__ == '__main__':
    unittest.main()
    
