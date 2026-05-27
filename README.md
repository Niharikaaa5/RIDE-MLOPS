# Ride Fare Prediction using MLOps

An end-to-end MLOps project for predicting ride fares using machine learning models, experiment tracking, data versioning, and API deployment.

## 🚀 Features

- Data preprocessing and feature engineering
- Distance calculation using geographic coordinates
- Regression model training and evaluation
- MLflow experiment tracking
- DVC for data and pipeline versioning
- FastAPI deployment for real-time predictions
- Reproducible ML pipeline

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- DVC
- FastAPI

## 🤖 Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## 📊 Evaluation Metrics

- R² Score
- RMSE (Root Mean Square Error)

## 📈 Results

Linear Regression achieved the best performance with an R² score of approximately 0.80 and the lowest RMSE after effective feature engineering.

## 📂 Project Workflow

1. Data Collection  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation  
6. Experiment Tracking using MLflow  
7. Data Versioning using DVC  
8. Deployment using FastAPI  

## ▶️ Run the Project

```bash
git clone https://github.com/your-username/Ride-Fare-Prediction-MLops.git

cd Ride-Fare-Prediction-MLops

pip install -r requirements.txt

uvicorn app:app --reload
```

## 📌 Future Improvements

- Add XGBoost/CatBoost
- Docker integration
- CI/CD pipeline
- Kubernetes deployment
- Model monitoring
