# House-Price-Prediction-TechnoHacks
Welcome to the House Price Prediction project using Random Forest Regression! In this repository, we explore how to predict house prices based on various features using a Random Forest Regression model. Accurate price prediction is essential for real estate and property valuation.

Overview

In this project, we follow these steps to build a predictive model:
Data Loading: We begin by loading the dataset from a CSV file named tested.csv using the pandas library. The dataset contains information about houses, including features like square footage, number of bedrooms, location, and price.

Data Preprocessing: We preprocess the data to make it suitable for modeling. This includes converting the date to a usable format (e.g., extracting year, month, and day), dropping unnecessary columns (e.g., 'id' and 'date'), and handling categorical variables (e.g., one-hot encoding for 'zipcode').

Data Splitting: We split the data into training and testing sets using the train_test_split function from scikit-learn. This separation allows us to train the model on one portion of the data and evaluate its performance on another.

Feature Scaling: Standardization is applied to the features using the StandardScaler from scikit-learn. This ensures that all features have the same scale and avoids any feature dominating the others.

Random Forest Regression: We initialize a Random Forest Regressor and define a hyperparameter grid for hyperparameter tuning. The hyperparameters include the number of estimators, maximum depth, minimum samples split, and minimum samples leaf. Grid search with cross-validation is performed to find the best combination of hyperparameters.

Model Training: We train the Random Forest Regressor with the best hyperparameters obtained from grid search.

Model Evaluation: The model's performance is evaluated using several metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and cross-validated RMSE. Visualizations such as a scatter plot of predicted vs. actual prices, feature importance plot, residual plot, and distribution of residuals are also provided to understand the model's behavior and performance.

Usage

Clone this repository:
bash
Copy code
git clone https://github.com/your-username/house-price-prediction.git

Navigate to the project directory:
bash
Copy code
cd house-price-prediction
Ensure you have the tested.csv file in the same directory as the code.

Run the Jupyter Notebook or Python script to execute the project.

Requirements

Before running the code in this repository, ensure that you have the necessary Python libraries installed. You can install them using pip:
bash
Copy code
pip install pandas scikit-learn matplotlib
