import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
from stopwords import georgian_stopwords
import json
import pandas as pd
import requests
import html


"""
utils.py

This script is necessary for preparing the model for input by removing stopwords and performing additional preprocessing steps on the dataset.

1) **Tokenization**: We first tokenize the text by splitting it into an array where each index represents a word. For example:
    Example:
   "The quick brown fox jumps over the lazy dog" 
   becomes: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

2) **Removing Stopwords**: We then remove common stopwords (such as "the", "is", "and", etc.) which don't add meaningful information to the text.
   Example:
   ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] 
   becomes: ["quick", "brown", "fox", "jumps", "lazy", "dog"]

3) **Stemming**: In this step, we reduce each word to its root or base form by applying stemming algorithms. This ensures that variations of a word (like "jumps" and "jump") are treated as the same word.
   Example: 
   - "jumps" becomes "jump"
   - "running" becomes "run"

4) **Lemmatization**: We further refine the words by applying lemmatization. Unlike stemming, which may result in non-dictionary words, lemmatization converts words to their base forms based on their meanings.
   Example: 
   - "better" becomes "good"
   - "running" becomes "run" (but based on part of speech context)

5) **Output**: The final result is a list of cleaned and processed words ready for use in the model.
"""


""" This gets english and italian stopwords from our stopwords import. The stopwords are used to aid in filtering inputs by removing unnecessary words which improves accuracy and efficiency """
# store the stopwords in a list, exclude "not"
nltk.download('stopwords')
english_sw = stopwords.words('english') 
# add italian stopwords
italian_sw = stopwords.words('italian')

# Function to get the root directory of the project
def get_root_dir():
  return Path(__file__).parent.parent

# Function to remove hashtags, links, and mentions from a string
def remove_hashtags_links_mentions(text):
  text = re.sub(r'#\w+', '', text)  # Remove hashtags
  text = re.sub(r'http\S+', '', text)  # Remove links
  text = re.sub(r'@\w+', '', text)  # Remove mentions
  return text

# Convert text to lowercase
def format_lowercase_special_chars(text):
  text = text.lower()
  text = re.sub(r'[^a-zA-Z0-9/ \u0080-\uFFFF]', '', text)
  return text

# This function creates stopwords list excludes ['not']
def create_stopwords():
  # add georgian stopwords
  sw = english_sw + italian_sw + georgian_stopwords
  return sw

# This function creates tokens along with pos. By tokenizing the text, we are splitting the words up so stopwords can be removed
def create_tokens_with_pos(data, sw_list):
  twt_tkn_pos_list = list()
  # Check type of data
  if type(data) == str:
    data = [data]
  else:
    data = data["text"]

  for idx, twt in enumerate(data):
    tkn_list = word_tokenize(twt) # tokenize
    # remove stop words
    tkn_list = [word for word in tkn_list if word not in sw_list]
    pos_list = pos_tag(tkn_list) # get pos for each tkn
    twt_tkn_pos_list.append(pos_list) # save in a list

  print("sample token with pos conversion :", \
        f"\n {data[0]}", \
        f"\n {twt_tkn_pos_list[0]}")

  return twt_tkn_pos_list

'''
This function lemmatizes each word given word, pos_tag
Lemmatizing a word reduces a word to its basic form. For example, running becomes run, mice becomes mouse, etc.
'''
def lemmatize_word(word, tag):
  wnl = WordNetLemmatizer()
  wntag = tag[0].lower()
  wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
  lemma = wnl.lemmatize(word, wntag) if wntag else word
  return lemma

'''
This function lemmatizes tweets passeed as
list of list, where each item is a tuple
tuple -> (word, pos)
'''
def lemmatize_tweets(twt_tkn_pos_list):
  #lemmatize the word in tweets_to_token
  lemma_tkn_list = list()

  for tkn_list in twt_tkn_pos_list:
    temp_list = list()
    for word, tag in tkn_list:
      temp_lemma = lemmatize_word(word, tag)
      temp_list.append(temp_lemma)
    lemma_tkn_list.append(str(temp_list))
  
  print(f"\n sample lemmatization output:")
  print(f"twt_tkn \t\t lma_tkn")
  for twt_tkn, lma_tkn in zip(twt_tkn_pos_list[0], lemma_tkn_list[0]):
    print(twt_tkn, f"\t\t", lma_tkn)
  
  return lemma_tkn_list

'''
This function add a new column - ["lemma_txt"] 
to the dataframe with lemmatized tokens 
'''
def add_lemmatization(data):    
  print('\n Inside "add_lemmatization" function')
  num_elements = 10
  # create stopwords list
  stopwords_list = create_stopwords()
  # create tokens along with pos
  twt_token_pos_list = create_tokens_with_pos(data, stopwords_list)
  # lemmatize the tweets stored as list of list [[word,pos]]
  lemma_token_list = lemmatize_tweets(twt_token_pos_list)
  if type(data) == str:
    return lemma_token_list
  
  # add a new coumn to the tweet_df with the lemma_token_list
  data = data.assign(lemma_txt=lemma_token_list)
  # re-order columns to ['text', 'lemma_txt', 'classification']
  data = data[data.columns[[0,2,1]]] 
  print(f"\n DF after performing lemmatization:  \n\n{data.tail(num_elements)}")
  return data

"""
Apply stemming to a string.
Stemming is the process of reducing a word to its root form by removing suffixes and other word endings
content: string to apply stemming to
returns: stemmed string
"""
def stemmer(data):
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')
    stop_words.extend(stopwords.words('italian'))
    words = re.findall(r'\b\w+\b', data.lower())
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    stemmed_content = ' '.join(stemmed_words)
    return stemmed_content

def translate_tweets(in_file_path, out_file_path, target):
  # NOTE THAT SOME OF THE SYNTAX MAY NEED TO BE CHANGED FOR SPECIFIC USE
  # Define Google Translate URL and API key
  GOOGLE_TRANSLATE_API_KEY = "AIzaSyBiSHDLVxUGbLudTKEYBo1-SNAJ1PgEj1I"
  GCP_TRANSLATE_URL = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_TRANSLATE_API_KEY}"  

  # Define excel sheet path and load into pandas
  df = pd.read_excel(in_file_path)

  # Extract the 'Tweet' column
  input_texts = df['Tweet'].tolist()
  translations = []

  # Create a new 'ID' column with unique IDs
  df['ID'] = range(1, len(df) + 1)

  # Iterate through the input texts and identifier column (ID)
  for input_text, tweet_id in zip(input_texts, df['ID']):
    data = {
        "target": target,
        "q": input_text,
    }

    headers = {
        "Content-Type": "application/json",
    }

    try:
        # Obtain API reponse
        # Replace &#39 with appropriate character
        response = requests.post(GCP_TRANSLATE_URL, json=data, headers=headers)
        if response.status_code == 200:
            translation_result = response.json()
            translations.append(translation_result)
            print(translation_result)
            translated_text = translation_result.get('data', {}).get('translations', [{}])[0].get('translatedText', '')
            # Remove or decode HTML entities using the html library
            translated_text = html.unescape(translated_text)
            # Set the unescaped text to the 'Tweet_Italian' column
            df.loc[df['ID'] == tweet_id, 'Translated_Tweet'] = translated_text
            print(translated_text, " at index {tweet_id}")
        else:
            print(f"Translation request failed for input: {input_text}")
            translations.append({})
    except requests.exceptions.Timeout:
        print(f"Translation request timed out for input: {input_text}")
        break 

  # Save the DataFrame back to the Excel file with the new column
  df.to_excel(out_file_path, index=False)

  return df

# Define the hyperparameter grids for each model
lr_param_grid = {
    'C': [0.1, 0.5, 1.0, 10.0],
    'penalty': ['l1', 'l2', 'elasticnet']
}

mnb_param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'fit_prior': [True, False],
}

pac_param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
}

bnb_param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'binarize': [0.0, 0.5, 1.0],
    'fit_prior': [True, False]
}

svc_param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0],
    'loss': ['hinge', 'squared_hinge'],
    'penalty': ['l1', 'l2'],
    'max_iter': [500, 1000, 2000]
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

dtc_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

mlp_param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': np.logspace(-5, 3, 9),
    'learning_rate': ['constant', 'adaptive']
}

# Dictionary containing the different models and their corresponding hyperparamater grids. We have this so we can test each one to find the bes tone
models = {
    'Logistic Regression': {
        'model': LogisticRegression,
        'params': lr_param_grid
    },
    'MultinomialNB': {
        'model': MultinomialNB,
        'params': mnb_param_grid
    },
    'PassiveAggressiveClassifier': {
        'model': PassiveAggressiveClassifier,
        'params': pac_param_grid
    },
    'BernoulliNB': {
        'model': BernoulliNB,
        'params': bnb_param_grid
    },
    'LinearSVC': {
        'model': LinearSVC,
        'params': svc_param_grid
    },
    # 'RandomForestClassifier': {
    #     'model': RandomForestClassifier,
    #     'params': rf_param_grid
    # },
    # 'DecisionTreeClassifier': {
    #     'model': DecisionTreeClassifier,
    #     'params': dtc_param_grid
    # }
}
