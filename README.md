# House-Price-Predictor
Fundamentals of AI and ML Evaluated Course Project

# Overview of the project
This is a **Machine Learning system** that predicts house prices from size (sqft) using **NumPy linear regression**. The model learns pricing patterns from real data and deploys as a portable CSV file. Users can train the model, save it, and predict prices for any house size instantly.

# Features
1.) Trains linear regression model on house size-price data using Normal Equation.<br>
2.) Predicts prices for new houses (ex: 2500sqft is Rs 30 Lakhs) .<br>
3.) Saves trained model to CSV with parameters + performance metrics.<br>
4.) Loads model from CSV for instant production predictions.<br>
5.) Zero external ML dependencies (NumPy and pandas).<br>
6.) Rs 0.25Lakhs accuracy on Rs 19-45Lakhs price range.<br>

# Technologies/Tools used
Language - Python 3<br>
ML Core - NumPy<br>
Data - Pandas<br>
Deployment - CSV persistence<br>

# Steps to Install & Run the Project
1.) Ensure Python 3.8+, NumPy, Pandas installed (`pip install numpy pandas`).<br>
2.) Save code as `predictor.py`.<br>
3.) Open terminal, navigate to file directory.<br>
4.) Run: `python predictor.py`.<br>
5.) Model trains automatically and saves `house_price_model.csv`.<br>

# Instructions for testing
1.) Run `python predictor.py` - verify model trains and saves CSV.<br>
2.) Check `house_price_model.csv` contains intercept(16 Lakhs), slope(0.046), predictions.<br>
3.) Test predictions: Edit code with `predict_from_csv(2200)` this should return - 28 Lakhs .<br>
4.) Delete CSV and rerun - model retrains successfully.<br>
5.) Test edge cases: 1000sqft, 5000sqft predictions.<br>
6.) Verify MAE <Rs 0.5 Lakhs in results DataFrame.<br>
