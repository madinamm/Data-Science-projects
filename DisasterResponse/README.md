# Disaster Response Pipeline Project

### Project Summary
The main purpose of this project is to analyze messages in social media during disasters, process and classify those message into 36 categories (e.g. Earthquake, Fire, Floods etc.) so that the relevant disaster relief agencies will be informed and react on-time. The final output is an app, which classifies messages into the relevant categories based on the inputted message and can be used by an employee from a disaster relief agency. In this project for training purposes, disaster messages from social media in the past and categories were used.
The project consists of 3 parts:
- ETL Pipeline that cleans original messages and categories csv-s used for training purposes and stores them in a database;
- ML Pipeline that trains a classifier;
- final user-friendly app, where any disaster message can be inputted and categorization will be performed as per above steps.

The database used was provided by Appen (formerly Figure 8) to Udacity Nanodegree students. This project was built with the support and as part of my learning journey within Udacity Nano degree program.

### Instructions:

General Note: all the files should be placed in one folder, with one exception which has a folder: `template` folder with 2 html files.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python process_data.py messages.csv categories.csv DisasterResponse.db`
        where instead of DisasterResponse a different db name can be used (Table name = Message)
        For more information about ETL logic, please refer to a relevant NB: ETL Pipeline Preparation.ipynb

    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py DisasterResponse.db classifier.pkl`
      For more information, please refer to a relevant NB: ML Pipeline Preparation.ipynb
      Note: We used lower number of hyperparameter tunning to have better speed performance of a run, which can be adjusted as per need (those lines were commented out in the Notebook)
      
2. Run your web app: `python run.py`  (We do not have a separate folder)

3. Click the `PREVIEW` button to open the homepage

### Explanation of files:
All required files are in 1 folder, except HTML files required for the app, which are in the folder `templates`.
    Files required for ETL pipeline with training data:
    - messages.csv
    - categories.csv
    ETL Pipline and NB:
    - ETL Pipeline Preparation.ipynb
    - process_data.py
    Databse output from ETL Pipeline:
    - DisasterResponse.db
    Screenshot example of a successful pipeline:
    - 1. Screenshot - Data Load logs.png
    
    ML Pipeline and NB:    
    - ML Pipeline Preparation.ipynb
    - train_classifier.py
    Output from ML Pipeline in pkl file:
    - classifier.pkl

    App results:
    - run.py
    - `templates` folder:
      a. go.html
      b. master.html

### Acknowledgements
Dataset credits: Appen (formerly Figure 8) team. This Analysis was made possible thanks to the courtesy of Appen in sharing data and a content of Udacity Nanodegree program.