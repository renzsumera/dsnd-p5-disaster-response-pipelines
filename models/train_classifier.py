# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import sqlite3
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    ''' Takes the database file path to load data. Runs a query to extract the
    values for the X, Y & category names and returns them.'''

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # connect to the database
    conn = sqlite3.connect(database_filepath)

    # run a query
    df = pd.read_sql('SELECT * FROM disaster', conn)

    # extract values for X, Y and category names
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(Y)

    return X, Y, category_names

def tokenize(text):
    '''Uses NLTK to case normalize, lemmatize, and tokenize text. This function
    is used in the machine learning pipeline to vectorize and apply TF-IDF to
    the text.'''

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''Builds the pipeline that processes text and then performs multi-output
    classification on the 36 categories in the dataset. GridSearchCV is used to
    find the best parameters for the model.'''

    # build pipeline
    pipeline = Pipeline([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])

    # use grid search to find better parameters
    parameters = {
         'text_pipeline__tfidf__use_idf': (True, False),
         'clf__estimator__n_estimators': [10, 50, 100],
         'clf__estimator__learning_rate': [0.1, 1, 5],
         }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Uses the model to predict the classification of the 36 categories.
    Reports the F1 score, precision, and recall for each output category of the
    dataset.'''

    # predict on test data
    Y_pred = model.predict(X_test)

    # report the f1 score, precision, and recall for each category
    for i in range(36):
        category = category_names[i]
        f1 = f1_score(Y_test.iloc[:,i], Y_pred[:,i])
        precision = precision_score(Y_test.iloc[:,i], Y_pred[:,i])
        recall = recall_score(Y_test.iloc[:,i], Y_pred[:,i])
        print(category)
        print("\tF1 Score: %.4f\tPrecision: %.4f\t Recall: %.4f\n" % (f1, precision, recall))

def save_model(model, model_filepath):
    '''Stores the classifier into a pickle file to the specified model file
    path.'''

    # export model as pickle file
    pickle.dump(model, open(model_filepath, 'wb'))

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
