import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    """ Load disaster messages and categories into dataframe.

    Args:
        messages_filepath: String. This is a csv file and contains disaster messages.
        categories_filepath: String. This is a csv file and contains disaster categories for each messages.

    Returns:
       pandas.DataFrame

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on = 'id', how = 'outer')
    return df




def clean_data(df):
    """Clean data.
    Args:
        df: pandas.DataFrame. it contains disaster messages and categories.

    Return:
        pandad.DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1]).tolist()

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace = True)
    return df
    


def save_data(df, database_filename):
    """Save DataFrame into database.

    Args:
        df: pandas.DataFrame. 
        database_filename: String. Dataframe is saved into this database file.
    """    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('CleanedDataTable', engine, if_exists='replace',index=False)
    pass  


def main():
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
        print('No files found')


if __name__ == '__main__':   
    main()