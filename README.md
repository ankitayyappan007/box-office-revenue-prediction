Box Office Movie Revenue Prediction (Machine Learning Project)

Predicting movie revenue and ROI (Return on Investment) using machine learning techniques and metadata from the TMDB movie dataset. This project includes end-to-end data preprocessing, feature engineering, model training, evaluation, and deployable scripts.

Overview of the Project

This project builds a machine learning model capable of predicting movie revenue using features such as:

Budget

Genres

Cast popularity

Production companies

Runtime

Keywords

Release date

Popularity

Using Python and scikit-learn, the project transforms raw movie metadata into structured numerical features suitable for ML models.

A full Jupyter Notebook is included with Exploratory Data Analysis (EDA), visualizations, and complete ML workflow.

Project Layout
├── app.py                              # Script for making predictions using the trained model
├── train_model.py                      # Script to train ML models
├── BoxOffice_ROI_FullyUpdated.ipynb    # Jupyter Notebook with EDA + ML pipeline
├── tmdb_5000_movies.csv                # Dataset 1
├── tmdb_5000_credits.csv               # Dataset 2
├── Dockerfile                          # Containerization for deployment
├── model/                              # Trained model files (after training)

└── README.md

Machine Learning Workflow

1. Data Preprocessing

Removed missing values

Converted JSON-like fields (genres, companies, cast)

Cast Popularity Extracted

Preparing ROI-based features

Merged movie and credit datasets

Normalized budget & revenue

2. Feature Engineering

One-hot encoding for genre

Number of production companies

Cast score based on popularity

Normalization at runtime

Extraction of year of release

3. Model Training

Trained various supervised ML models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

4. Model Evaluation

Metrics used:

R² Score

Mean Absolute Error MAE

MSE represents Mean Squared Error.
Random Forest yielded the best performance.

Running the Project

1. Install Dependencies

pip install -r requirements.txt

2. Training the Model

python train_model.py


3. Execute Predictions Application

python app.py

Run Using Docker

Build the image:

docker build -t movie-revenue-predictor.

Execute the container:

docker run -p 8000:8000 movie-revenue-predictor

Project Outcomes

The model identifies major factors influencing movie revenue:

High budgets

Strong popularity of the leading cast.

Popular genres

Successful production companies

Release trends per year

Movie popularity score

Random Forest yielded the most exact predictions.

Skills Demonstrated

Machine Learning: Regression Models

Data Cleaning & Feature Engineering

Exploratory Data Analysis (EDA)

Python (pandas, numpy, sklearn)
Jupyter Notebook Docker Deployment Creation of an end-to-end ML pipeline Authors Ankit Subhash Ayyappan B.Tech CSE (AI & Data Science), IIIT Kottayam Email: ankitayyappan007@gmail.com