import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = 'featured_train_data.csv'
TARGET_COLUMN = 'price'
LEAKY_COLUMNS = ['median_sale_price', 'median_list_price', 'Median_Home_Value']


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def summarize_data(df):
    print("Dataset shape:", df.shape)
    print("Basic Statistics:")
    return df.describe()


def plot_price_distribution(df, target=TARGET_COLUMN):
    plt.figure(figsize=(8, 4))
    plt.hist(df[target], bins=50, color='steelblue', edgecolor='black')
    plt.title('House Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.show()


def clean_data(df, drop_columns=LEAKY_COLUMNS):
    return df.drop(columns=drop_columns).dropna()


def split_features_target(df, target=TARGET_COLUMN):
    return df.drop(target, axis=1), df[target]


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.show()


def main(path=DATA_PATH):
    df = load_data(path)
    summarize_data(df)
    plot_price_distribution(df)

    X, y = split_features_target(clean_data(df))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    print("Model trained successfully!")

    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"\nA R² of {r2:.2f} means our model explains "
          f"{r2*100:.0f}% of the price variation")

    plot_actual_vs_predicted(y_test, model.predict(X_test))
    return model, mse, r2


if __name__ == '__main__':
    main()
