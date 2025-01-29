import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes and lemmatizes the input text.

    Args:
        text (str): The text to be tokenized and lemmatized.

    Returns:
        list: A list of clean tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Renders the index webpage displaying visuals and receiving user input.

    Extracts data needed for visuals and creates graphs to be displayed on the webpage.
    
    Returns:
        str: Rendered HTML template for the index page.
    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Calculate the sum of boolean values for the specified categories
    category_columns = [
        'related', 'request', 'offer', 'aid_related', 'medical_help',
        'medical_products', 'search_and_rescue', 'security', 'military',
        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
        'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake',
        'cold', 'other_weather', 'direct_report'
    ]
    
    category_sums = df[category_columns].sum().sort_values(ascending=True)
    category_names = list(category_sums.index)

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    y=category_names,
                    x=category_sums,
                    orientation='h'  # horizontal bar chart
                )
            ],
            'layout': {
                'title': 'Messages per Category',
                'xaxis': {
                    'title': "Count"
                },
                'yaxis': {
                    'title': "Categories",
                    'ticklabelposition': "inside"
                },
                'height': 1000
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hole=0.3  # Optional: creates a donut chart
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres (Pie Chart)',
                'height': 600
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres (Bar Chart)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handles user query and displays model results.

    Retrieves user input, predicts the classification using the model,
    and renders the results on the go.html page.

    Returns:
        str: Rendered HTML template for the go page with query results.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Runs the Flask application.

    Starts the Flask web server on host '0.0.0.0' and port 3000 in debug mode.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()