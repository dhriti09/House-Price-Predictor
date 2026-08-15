from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from utils import (
    evaluate_model,
    load_dataset,
    plot_actual_vs_predicted,
    plot_histogram,
    prepare_features,
)

LEAKY_COLUMNS = ['median_sale_price', 'median_list_price', 'Median_Home_Value']

df = load_dataset('featured_train_data.csv')
df.head()

print("Basic Statistics:")
df.describe()

plot_histogram(df['price'], 'House Price Distribution', 'Price')

X, y = prepare_features(df, target='price', drop_columns=LEAKY_COLUMNS)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred)

plot_actual_vs_predicted(y_test, y_pred)
