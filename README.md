# **Car Price Prediction Using Machine Learning**

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Modeling](#modeling)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyper-tuning](#hyper-tuning)
- [Results](#results)
- [License](#license)

---

## **Project Overview**
This project aims to build a predictive model to accurately estimate car prices in Australia using historical data and machine learning as well as neural network techniques. The goal is to build the model using Python, Scikit-learn, and other libraries to predict car prices from a given dataset. It demonstrates the process of data preprocessing, feature engineering, model training, and evaluation.

---

## **Dataset**
The dataset used for this project contains information about used and new cars, including the following features:
- **Brand**: Name of the car manufacturer.
- **Year**: Year of manufacture or release.
- **Model**: Name or code of the car model.
- **Car/Suv**: Type of the car (car or SUV) grouped by region and supplier.
- **Title**: Title or description of the car.
- **UsedOrNew**: Condition of the car (used, new, demo).
- **Transmission**: Type of transmission (manual or automatic).
- **Engine**: Engine capacity or power with cylinders (in litres).
- **DriveType**: Type of drive (front-wheel, rear-wheel, all-wheel, etc.).
- **FuelType**: Type of fuel (petrol, diesel, hybrid, electric, etc.).
- **FuelConsumption**: Fuel consumption rate (in litres per 100 km).
- **Kilometres**: Distance travelled by the car (in kilometres).
- **ColourExtInt**: Colour of the car (exterior and interior).
- **Location**: Location of the car (city and state).
- **CylindersinEngine**: Number of cylinders in the engine.
- **BodyType**: Shape of the car body (sedan, hatchback, coupe, etc.).
- **Doors**: Number of doors in the car.
- **Seats**: Number of seats in the car.
- **Price**: Price of the car (in Australian dollars).


### **Dataset Source**
https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices/data

---

## **Dependencies**
This project requires the following Python libraries:
- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical computations.
- `matplotlib`: Data visualization.
- `seaborn`: Statistical data visualization.
- `sklearn`: Machine learning algorithms and tools.
- `tensorflow`: Deep learning algorithms.
- `xgboost`: XGB model development
- `datetime`: Feature engineering

To install these dependencies, the following commands can be used:
```bash
pip install -r requirements.txt
```

---

## **Modeling**
The machine learning models used for this prediction include:
- **Linear Regression**: A simple model to predict continuous values.
- **Decision Tree**: A tree model to capture nonlinear relationships.
- **K-Nearest Neighbour (KNN)**: A model for capturing local patterns.
- **Random Forest Regressor**: A tree-based ensemble method.
- **Gradient Boosting Regressor**: An ensemble method for better performance.
- **XGBoost**: A powerful gradient boosting algorithm.

The deep learning models used for in this project include:
- **Gated Recurrent Unit (GRU)**: A type of recurrent neural network (RNN) that enhances the speed performance of LSTM networks by simplifying the structure with only two gates: the update gate and the reset gate.
- **Multilayer Perceptron (MLP)**: A fully connected feedforward neural network that maps input features to continuous target values.
- **Long Short-Term Memory (LSTM)**: A type of recurrent neural network (RNN) designed to capture long-term dependencies.
- **Recurrent Neural Network (RNN)**: A neural network that captures dependencies in sequential data by passing the hidden state from one time step to the next.

---

## **Evaluation Metrics**
ML models are evaluated using the following metrics:
- **R-Squared (R²)**
- **Adjusted R²**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error Percentage (MAPE)**

DL models are evaluated using the following metrics:
- **R-Squared (R²)**
- **Loss**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

---

## **Hyper-tuning**
The following parameters were selected for respective models using 5-fold cross-validation of `GridSearchCV()` to find the best R²:
- **Decision Tree**: {`criterion`, `friedman_mse`, `max_depth`, `min_samples_leaf`, `min_samples_split`}
- **KNN**: {`metric`, `manhattan`, `n_neighbors`, `weights`}
- **Random Forest**: {`bootstrap`, `max_depth`, `max_features`, `min_samples_leaf`, `min_samples_split`, `n_estimators`}
-  **Gradient Boosting**: {`learning_rate`, `max_depth`, `max_features`, `min_samples_leaf`, `min_samples_split`, `n_estimators`, `subsample`}
-  **XGB**: {`learning_rate`, `max_depth`, `n_estimators`, `subsample`}


---

## **Results**
The model results are summarized below:
- **Best model**: XGB with an R² score of 0.78.
- **Visualizations**: The notebook includes various plots to show the relationship between features and the target variable (price).
- **Correlation**:
  - High positive correlation among: *‘EngineSize’*, *‘CylinderinEngine’*, *‘FuelPer100km’*.
  - High negative correlation between: *‘Year’* and *‘Kilometre’*, *‘Kilometres’* and *‘Price’*.
- **Feature Importance**: The top 3 features found to be : *DriveType_Front*, *CylindersinEngine* and *Year*.

---

## **License**
This project is licensed under the MIT License - see the [LICENSE] file for details.
