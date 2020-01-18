# Disaster Response Pipeline Project

### Background
Disaster Response Pipeline Project is to show Data Engineering skills, including ETL and ML Pipeline preparation and creating model in a data visualisation web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

### Important Files:
- `data/process_data.py`: The ETL pipeline used to process data in preparation for ML model building.
- `models/train_classifier.py`: The Machine Learning pipeline used to fit, evaluate, and save the model to a Python pickle file (pickle is not uploaded to the Github due to size constraints.).
- `app/run.py`: Run the web visualizations app.

### Results
1. An ETL pipleline was built to read data, clean data, and save data into a SQLite database.
2. A ML pipepline was developed to train a classifier to performs multi-output classification.
3. A Flask app was created to show data visualization.
