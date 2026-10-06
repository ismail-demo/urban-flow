import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Load data
df = pd.read_csv('transit_data.csv')

X = df[['time_of_day', 'day_of_week', 'transit_line', 'weather']]
y_delay = df['Delay_Minutes']
y_crowding = df['Crowding_Level']

# Preprocessor to one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['time_of_day', 'day_of_week', 'transit_line', 'weather'])
    ])

# Pipeline for Delay Model (Multiple Linear Regression)
delay_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Pipeline for Crowding Model (Bagging Classifier)
crowd_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', BaggingClassifier(n_estimators=10, random_state=42))
])

# Train Delay Model
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_delay, test_size=0.2, random_state=42)
delay_pipeline.fit(X_train_d, y_train_d)
y_pred_d = delay_pipeline.predict(X_test_d)
print(f"Delay Model RMSE: {mean_squared_error(y_test_d, y_pred_d) ** 0.5:.2f}")

# Train Crowding Model
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crowding, test_size=0.2, random_state=42)
crowd_pipeline.fit(X_train_c, y_train_c)
y_pred_c = crowd_pipeline.predict(X_test_c)
print(f"Crowding Model Accuracy: {accuracy_score(y_test_c, y_pred_c):.2f}")

# Save models
with open('delay_model.pkl', 'wb') as f:
    pickle.dump(delay_pipeline, f)

with open('crowd_model.pkl', 'wb') as f:
    pickle.dump(crowd_pipeline, f)

print("Models saved as delay_model.pkl and crowd_model.pkl")
