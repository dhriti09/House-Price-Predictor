import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load your dataset
df = pd.read_csv('featured_train_data.csv')

print("Dataset shape:", df.shape)
df.head()

print("Basic Statistics:")
df.describe()

plt.figure(figsize=(8,4))
plt.hist(df['price'], bins=50, color='steelblue', edgecolor='black')
plt.title('House Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

# Drop non-numeric or irrelevant columns
df_clean = df.drop(columns=['median_sale_price', 'median_list_price', 
                              'Median_Home_Value'])

# Handle missing values
df_clean = df_clean.dropna()

# Features and target
X = df_clean.drop('price', axis=1)
y = df_clean['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"\nA R² of {r2:.2f} means our model explains {r2*100:.0f}% of the price variation")

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()