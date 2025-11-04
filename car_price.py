




import pandas as pd
import numpy as np
df=pd.read_csv("/content/car data (2).csv")
df

a=df["Seller_Type"].nunique()
b=df["Seller_Type"].value_counts()
c=df["Seller_Type"].unique()
print(a)
print(b)
print(c)

from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the dataset

# Step 1: Drop 'Car_Name' column
df.drop(columns=['Car_Name'], inplace=True)

# Step 2: Convert 'Year' into 'Car_Age'
current_year = datetime.now().year
df['Car_Age'] = current_year - df['Year']
df.drop(columns=['Year'], inplace=True)  # Drop original 'Year' column

# Step 3: Encode categorical variables
le = LabelEncoder()
df['Seller_Type'] = le.fit_transform(df['Seller_Type'])  # Label Encoding for Seller_Type
df['Transmission'] = le.fit_transform(df['Transmission'])  # Label Encoding for Transmission
df = pd.get_dummies(df, columns=['Fuel_Type'], drop_first=False)  # One-Hot Encoding for Fuel_Type

# Display the updated dataset structure
print(df.info())
print(df.head())
from sklearn.preprocessing import MinMaxScaler
# Step 4: Scale 'Kms_Driven'
scaler = MinMaxScaler()
df['Kms_Driven'] = scaler.fit_transform(df[['Kms_Driven']])

# Outlier Handling using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    # Removing outliers for numerical features
df = remove_outliers(df, 'Selling_Price')
df = remove_outliers(df, 'Present_Price')
df = remove_outliers(df, 'Kms_Driven')
df = remove_outliers(df, 'Car_Age')

import matplotlib.pyplot as plt
import seaborn as sns
# Exploratory Data Analysis (EDA)
# Statistical Summary
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Distribution of Numerical Features
df.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# Boxplots for Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient="h")
plt.title("Boxplots for Outlier Detection")
plt.show()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Split Data into Training and Testing Sets
X = df.drop(columns=['Selling_Price'])  # Features
y = df['Selling_Price']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train a Baseline Model (Linear Regression)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#  Model Evaluation
y_pred = lr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

# Print model performance
print("Linear Regression Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

print(f"R² Score: {r2:.2f}")
from sklearn.ensemble import RandomForestRegressor
#  Train a Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#  Model Evaluation
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

# Print model performance
print("Random Forest Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

print(f"R² Score: {r2:.2f}")

import joblib

joblib.dump(rf_model,"car_price_model.pkl")
joblib.dump(scaler,"scaler.pkl")

! pip install streamlit -q

import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Car Price Prediction 🚗💰")
st.write("Enter car details to predict the selling price.")

# User input fields

present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, format="%.2f")
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
car_age = st.number_input("Car Age (in years)", min_value=0)
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

# Encode categorical values
seller_type = 1 if seller_type == "Individual" else 0
transmission = 1 if transmission == "Automatic" else 0
fuel_type_cng = 1 if fuel_type == "CNG" else 0
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

# Scale 'Kms_Driven'
kms_driven_scaled = scaler.transform(np.array([[kms_driven]]))[0][0]

# Prepare input for prediction
features = np.array([[present_price, kms_driven_scaled, owner, seller_type, transmission, car_age, fuel_type_cng, fuel_type_diesel, fuel_type_petrol]])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Selling Price: ₹{prediction:.2f} Lakhs")

!wget -q -O - ipv4.icanhazip.com

! streamlit run app.py & npx localtunnel --port 8501
