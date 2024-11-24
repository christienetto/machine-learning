import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd

df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df.drop(columns='ID', inplace=True)  
X = df.drop(columns='log_pSat_Pa')  
X['parentspecies'] = X['parentspecies'].fillna(X['parentspecies'].mode()[0])  
X = pd.get_dummies(X, columns=['parentspecies'], drop_first=False)  
y = df['log_pSat_Pa']

X_sample = X[:1000]  
y_sample = y[:1000]

degrees = [1, 2, 3, 4, 5]

best_degree = 0
best_score = float('inf')

for degree in degrees:
    print("degree: ",degree)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring='neg_mean_squared_error')
    
    mean_score = -np.mean(scores) 
    print(mean_score)
    if mean_score < best_score:
        best_score = mean_score
        best_degree = degree

print(f"The best degree is {best_degree} with a mean square error of {best_score}")
