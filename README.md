# Disaster Response Pipeline project udacity

### Project Overview

This project analyze information from a real data set that contains real messages that were sent during disaster events. To analize the information is create a machine learning pipeline to categorize these events, to understand more easily which agency to call. The project the project has 3 steps: ETL to clean the data , Machine Learning to create the model and web app to make visualizations.

### Contents in the repository:

### ETL
*data/process_data.py

the file contains the code to do the following:
        Load the datasets.
        Merge the datasets.
        Clean the data.
        Store it in a SQLite database.
 
 ### Model
 *models/train_classifier.py
 
 the file contains the code to do the following:
 
        Load information from the SQLite database.
        Create train and test datasets
        Train and tune a machine learning model using GridSearchCV.
        Display the results on the test set.
        Create a pickle file with the model.
        
### app
* app/run.py

        Show the results in a Flask web app.
        
### Instructions: (taken from the original course file)
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
        
