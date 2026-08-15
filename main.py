import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path('featured_train_data.csv')
TARGET = 'price'
LEAKY_COLUMNS = ['median_sale_price', 'median_list_price', 'Median_Home_Value']


def load_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download it from "
            "https://www.kaggle.com/datasets/sarveshdhond/house-price-predictor-dataset "
            "and place it in the project root."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset '{path}' is not valid CSV: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset '{path}' contains no rows.")
    if TARGET not in df.columns:
        raise ValueError(
            f"Dataset '{path}' is missing the target column '{TARGET}'. "
            f"Found columns: {sorted(df.columns)}"
        )
    return df


def prepare_features(df):
    missing = [column for column in LEAKY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df_clean = df.drop(columns=LEAKY_COLUMNS).dropna()
    if df_clean.empty:
        raise ValueError(
            "No rows left after dropping missing values; the dataset has a "
            "missing value in every row."
        )

    X = df_clean.drop(TARGET, axis=1)
    y = df_clean[TARGET]

    non_numeric = X.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Features must be numeric for LinearRegression, but these are not: {non_numeric}"
        )
    if len(df_clean) < 2:
        raise ValueError(
            f"Need at least 2 rows to train and test, got {len(df_clean)}."
        )
    return X, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if len(X_test) == 0:
        raise ValueError(
            f"Test split is empty for a dataset of {len(X)} rows; provide more data."
        )

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"\nA R² of {r2:.2f} means our model explains {r2 * 100:.0f}% of the price variation")
    return y_test, y_pred


def plot_price_distribution(df):
    plt.figure(figsize=(8, 4))
    plt.hist(df[TARGET], bins=50, color='steelblue', edgecolor='black')
    plt.title('House Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.show()


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else DATA_PATH

    df = load_dataset(path)
    print("Dataset shape:", df.shape)
    print("Basic Statistics:")
    print(df.describe())

    plot_price_distribution(df)

    X, y = prepare_features(df)
    y_test, y_pred = train_and_evaluate(X, y)

    plot_actual_vs_predicted(y_test, y_pred)


if __name__ == '__main__':
    try:
        main(sys.argv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
