## lasso model is choosen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('data/studentperformance.csv')

df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])

df_dummies = pd.get_dummies(df)*1

y = df_dummies['Exam_Score']
X = df_dummies.drop('Exam_Score', axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


alpha_grid = {'alpha': np.logspace(-4, 4, 50)}

lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(lasso, alpha_grid, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train, y_train)




# Save model and scaler
joblib.dump(lasso_cv.best_estimator_, 'lasso_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl') 
