import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg')

import main  # noqa: E402


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    size = 60
    sqft = rng.uniform(500, 4000, size)
    return pd.DataFrame({
        'sqft': sqft,
        'homes_sold': rng.integers(1, 50, size),
        'median_sale_price': rng.uniform(1e5, 9e5, size),
        'median_list_price': rng.uniform(1e5, 9e5, size),
        'Median_Home_Value': rng.uniform(1e5, 9e5, size),
        'price': 200.0 * sqft + 50_000,
    })


@pytest.fixture
def clean_df(raw_df):
    return main.clean_data(raw_df)


@pytest.fixture(autouse=True)
def no_blocking_show(monkeypatch):
    monkeypatch.setattr(main.plt, 'show', lambda *a, **k: None)
    yield
    main.plt.close('all')


def test_load_data_reads_csv(tmp_path, raw_df):
    path = tmp_path / 'data.csv'
    raw_df.to_csv(path, index=False)

    loaded = main.load_data(str(path))

    assert list(loaded.columns) == list(raw_df.columns)
    assert len(loaded) == len(raw_df)


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        main.load_data(str(tmp_path / 'missing.csv'))


def test_summarize_data_returns_describe(raw_df, capsys):
    described = main.summarize_data(raw_df)

    assert 'mean' in described.index
    assert 'Dataset shape:' in capsys.readouterr().out


def test_clean_data_drops_leaky_columns(raw_df):
    cleaned = main.clean_data(raw_df)

    assert not set(main.LEAKY_COLUMNS) & set(cleaned.columns)
    assert 'price' in cleaned.columns
    assert len(cleaned) == len(raw_df)


def test_clean_data_drops_rows_with_missing_values(raw_df):
    raw_df.loc[0, 'sqft'] = np.nan

    cleaned = main.clean_data(raw_df)

    assert len(cleaned) == len(raw_df) - 1
    assert not cleaned.isna().to_numpy().any()


def test_clean_data_ignores_nan_in_dropped_columns(raw_df):
    raw_df.loc[0, 'median_sale_price'] = np.nan

    assert len(main.clean_data(raw_df)) == len(raw_df)


def test_clean_data_missing_drop_column_raises(raw_df):
    with pytest.raises(KeyError):
        main.clean_data(raw_df.drop(columns=['median_list_price']))


def test_clean_data_does_not_mutate_input(raw_df):
    before = raw_df.copy()

    main.clean_data(raw_df)

    pd.testing.assert_frame_equal(raw_df, before)


def test_split_features_target(clean_df):
    X, y = main.split_features_target(clean_df)

    assert 'price' not in X.columns
    assert y.name == 'price'
    assert len(X) == len(y) == len(clean_df)


def test_split_features_target_custom_target(clean_df):
    X, y = main.split_features_target(clean_df, target='sqft')

    assert 'sqft' not in X.columns
    assert y.name == 'sqft'


def test_train_model_fits_linear_regression(clean_df):
    X, y = main.split_features_target(clean_df)

    model = main.train_model(X, y)

    assert isinstance(model, LinearRegression)
    assert model.coef_.shape == (X.shape[1],)


def test_train_model_recovers_linear_relationship(clean_df):
    X, y = main.split_features_target(clean_df)

    model = main.train_model(X, y)

    assert model.coef_[list(X.columns).index('sqft')] == pytest.approx(200.0)
    assert model.intercept_ == pytest.approx(50_000, rel=1e-6)


def test_evaluate_model_on_perfect_fit(clean_df):
    X, y = main.split_features_target(clean_df)
    model = main.train_model(X, y)

    mse, r2 = main.evaluate_model(model, X, y)

    assert mse == pytest.approx(0.0, abs=1e-6)
    assert r2 == pytest.approx(1.0)


def test_evaluate_model_penalizes_bad_model(clean_df):
    X, y = main.split_features_target(clean_df)
    model = main.train_model(X, y)
    model.coef_ = np.zeros_like(model.coef_)
    model.intercept_ = 0.0

    mse, r2 = main.evaluate_model(model, X, y)

    assert mse > 0
    assert r2 < 0


def test_plot_price_distribution_creates_figure(raw_df):
    main.plot_price_distribution(raw_df)

    ax = main.plt.gcf().axes[0]
    assert ax.get_title() == 'House Price Distribution'
    assert ax.get_xlabel() == 'Price'


def test_plot_actual_vs_predicted_creates_figure(clean_df):
    X, y = main.split_features_target(clean_df)
    y_pred = main.train_model(X, y).predict(X)

    main.plot_actual_vs_predicted(y, y_pred)

    ax = main.plt.gcf().axes[0]
    assert ax.get_title() == 'Actual vs Predicted House Prices'
    assert ax.get_ylabel() == 'Predicted Price'


def test_main_end_to_end(tmp_path, raw_df, capsys):
    path = tmp_path / 'featured_train_data.csv'
    raw_df.to_csv(path, index=False)

    model, mse, r2 = main.main(str(path))

    assert isinstance(model, LinearRegression)
    assert mse == pytest.approx(0.0, abs=1e-6)
    assert r2 == pytest.approx(1.0)
    assert 'Model trained successfully!' in capsys.readouterr().out
