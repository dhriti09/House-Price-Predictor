"""Shared helpers for loading data, plotting and evaluating models."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PLOT_COLOR = 'steelblue'


def show_plot(title, xlabel, ylabel):
    """Apply the shared figure labelling and render the current figure."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_histogram(values, title, xlabel, ylabel='Count', bins=50, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    plt.hist(values, bins=bins, color=PLOT_COLOR, edgecolor='black')
    show_plot(title, xlabel, ylabel)


def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs Predicted House Prices',
                             xlabel='Actual Price', ylabel='Predicted Price', figsize=(8, 5)):
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.3, color=PLOT_COLOR)
    limits = [y_true.min(), y_true.max()]
    plt.plot(limits, limits, 'r--', linewidth=2)
    show_plot(title, xlabel, ylabel)


def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    return df


def prepare_features(df, target, drop_columns=()):
    """Drop leaky/irrelevant columns and missing rows, then split off the target."""
    df_clean = df.drop(columns=list(drop_columns)).dropna()
    return df_clean.drop(target, axis=1), df_clean[target]


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"\nA R² of {r2:.2f} means our model explains {r2 * 100:.0f}% of the price variation")
    return mse, r2
