import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): Path to the messages CSV file.
        categories_filepath (str): Path to the categories CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """Clean the merged dataframe by splitting categories and converting values.

    Args:
        df (pd.DataFrame): Merged DataFrame containing messages and categories.

    Returns:
        pd.DataFrame: Cleaned DataFrame with separate category columns.
    """
    # Split the categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to binary
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]  # Get the last character
        categories[column] = categories[column].astype(int)  # Convert to integer
    
    # Drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows where the 'related' column is equal to 2 (as it is a binary column)
    df = df[df['related'] != 2]

    # Reset the index and add a new index column
    df.reset_index(drop=True, inplace=True)
    df['index'] = df.index  # Add a new column with the current index values
    
    return df

def save_data(df, database_filename):
    """Save the cleaned data to a database.

    Args:
        df (pd.DataFrame): Cleaned DataFrame to be saved.
        database_filename (str): Path to the database file.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Message', engine, index=False, if_exists='replace')

def main():
    """Main function to load, clean, and save data."""
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'messages.csv categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()