import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings


# === Load the data ===
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# === Preprocess the data ===
# Drop unnecessary columns
X = train_data.drop(['ID', 'log_pSat_Pa'], axis=1)  # Features
y = train_data['log_pSat_Pa']                     # Target variable

# One-hot encode categorical columns in both train and test data
X = pd.get_dummies(X, columns=['parentspecies'], drop_first=True)
X_test = pd.get_dummies(test_data.drop(['ID'], axis=1), columns=['parentspecies'], drop_first=True)

# Ensure train and test have the same features after encoding
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# === Train-Test Split for Evaluation ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Model Training ===
# Initialize Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate the Model ===
# Predict on validation set
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error on Validation Set: {mse}")

# Cross-Validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print(f"Cross-Validation Mean Squared Error: {cv_mse}")


predictions = model.predict(X_test_scaled)


submission = pd.DataFrame({
    'ID': test_data['ID'],
    'TARGET': predictions
})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' has been created!")
