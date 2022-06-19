import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
       Function:
       load data 
       Args:
       database_filepath: path of database
       Return:
       X (DataFrame) : features dataframe
       Y (DataFrame) : target dataframe
       category (list) : target labels list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_table', engine)
    X = df['message']  # Message Column
    Y = df[df.columns[4:]] # Classification label
    return X, Y


def tokenize(text):
    """
    Tokenize the text function
    Args:
      text(str): Text message which needs to be tokenized
    Return:
      lemm(list of str): a list of the root form of the message words
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    stop = stopwords.words("english")
    words = [g for g in words if g not in stop]

    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(j) for j in words]

    return lemm


def build_model():
    '''
    build_model
    create the ML model with the informacion of database
    returns:
    model the definition of de ML model
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # set parameters
    parameters = { 'clf__estimator__n_estimators': [10, 15, 20] }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model



def evaluate_model(model, X_test, Y_test):
    """
    Function: Evaluate the model and print the metrics.
    model:  classification model
    X_test: messages
    Y_test: target
    """
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    '''
    save_model
    save model into a pickle archive
    input:
    model the parameters and definitions of the model
    model_filepath location of the model
    returns:
    pkl archive with the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
