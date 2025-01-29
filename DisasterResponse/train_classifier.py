import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('punkt')
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Load the data from the SQLite database.

    Args:
        database_filepath (str): The file path to the SQLite database.

    Returns:
        X (pd.Series): The feature data (messages).
        Y (pd.DataFrame): The target data (categories).
        category_names (list): The names of the categories.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre', 'index'])
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the input text.

    Args:
        text (str): The text to be tokenized.

    Returns:
        tokens (list): A list of cleaned and tokenized words.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens


def build_model():
    """
    Build a machine learning pipeline including a TfidfVectorizer and a 
    MultiOutputClassifier with XGBClassifier. This function also sets up 
    a GridSearchCV to perform hyperparameter tuning.

    Returns:
        grid_search (GridSearchCV): An instance of GridSearchCV configured 
        with the pipeline and hyperparameter grid for model training and 
        evaluation.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')))
    ])
    
    param_grid = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__max_depth': [6, 10]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model using the test data and print the classification report,
    confusion matrix, and accuracy for each category.

    Args:
        model: The trained model to be evaluated.
        X_test (pd.Series): The test feature data.
        Y_test (pd.DataFrame): The test target data.
        category_names (list): The names of the categories.
    """
    Y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(f"Classification report for {column}:")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        
        cm = confusion_matrix(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"Confusion Matrix:\n{cm}")
        
        accuracy = accuracy_score(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"Accuracy: {accuracy:.2f}\n")
        

def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Args:
        model: The model to be saved.
        model_filepath (str): The file path where the model will be saved.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Main function to execute the training of the classifier.
    It loads data, builds the model, trains it, evaluates it, and saves it.
    """
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