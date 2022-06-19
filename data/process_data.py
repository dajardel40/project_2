# import packages
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from two csv files and merge into a dataframe
    input:
    messages_filepath file of messages csv
    categories_filepath file of categories cvs
    returns:
    df the result of merge the two orginal files
    '''
  
    # read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge data
    df = messages.merge(categories, how='inner', on='id')
    return df

# clean data
def clean_data(df):
  '''
    clean_data
    clean the merge dataframe
    input:
    df the mergue dataframe
    returns:
    df the dataframe with new columns of 0 and 1 for each categorie and without duplicates
    '''
    # create categories
    categories = df['categories'].str.split(';', expand=True)
    # keep the first row of the categories dataframe
    row = categories.iloc[0,:]
  # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    #Convert category values to just numbers 0 or 1   
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'],axis = 1)
    # concatenate dataframes
    df = pd.concat([df,categories], axis = 1, join = 'inner' )
    df.drop_duplicates(inplace = True)
    return df
    
    
    # load to database
    
def save_data(df, database_filename):
    '''
    save_data
    save dataframe into a sqlite database
    input:
    df the clean dataframe
    database_filename name of database
    returns:
    db the result is the DisasterResponse database 
    '''
     engine=create_engine('sqlite:///' + database_filename)   
     df.to_sql('disaster_table', engine, index=False , if_exists='replace')

     # define features and label arrays


def main():
    if len(sys.argv) == 4:

       messages_filepath, categories_filepath, database_filepath =            sys.argv[1:]

       print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
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
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
