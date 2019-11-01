# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import matplotlib.pyplot as plt
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    """Load data from database into dataframe.

    Args:
        database_filepath: String. It contains cleaned data table.

    Returns:
       X: numpy.ndarray. data features
       Y: numpy.ndarray.  data categories
       category_name: list. Disaster category names.
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('CleanedDataTable', engine)

    X = df['message'].values
    y = df.iloc[:,4:].values
    category_names = df.columns[4:]
    return X, y, category_names


def tokenize(text, url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'):
    """Tokenize text

    Args:
        text: String. A disaster message
        url_regex: String. remove URLs

    Returns:
        tokens: list. It contains tokens.
    """
    # remove URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]",' ',text)
    
    #tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # Lemmatize
    tokens = [WordNetLemmatizer().lemmatize(i) for i in tokens]
    return tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        words = tokenize(text)
        pos_tags = nltk.pos_tag(words)
        if pos_tags !=[]:
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """Build model.

    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    # Set pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])

    # Set parameters for gird search
    parameters = {
         'features__text_pipeline__vect__max_df': (0.5,  1.0),
         'clf__estimator__n_estimators': [20, 50],
    }
    # Set grid search
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # Predict categories of messages.
    Y_pred = model.predict(X_test)    
    for i in range(Y_test.shape[1]): 
        print('*******',category_names[i],'*******')
        print(classification_report(Y_test[:,1], Y_pred[:,1]))
    pass

def save_model(model, model_filepath):
    """Save model

    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_filepath: String. Saved file path.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepaths')


if __name__ == '__main__':
    main()