# 🏠 House Price Predictor

A Machine Learning project that predicts house prices using real-world data with features like location, nearby amenities, economic indicators, and market trends.

Built using **Python** and **scikit-learn** as part of my journey into open source ML development.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| R² Score | 0.60 |
| Mean Squared Error | 34,614,131,073.93 |

> A R² of 0.60 means the model explains **60% of house price variation** using a simple Linear Regression — a strong baseline for real-world noisy data.

---

## 📦 Dataset
The dataset is not included in this repo due to file size.
Download it from: [House Price Predictor Dataset](https://www.kaggle.com/datasets/sarveshdhond/house-price-predictor-dataset)

## 📁 Dataset Features

The dataset includes a rich set of features:

**Market Data**
- `median_sale_price`, `median_list_price`, `median_ppsf`
- `homes_sold`, `pending_sales`, `inventory`, `median_dom`
- `avg_sale_to_list`, `sold_above_list`, `off_market_in_two_weeks`

**Location & Amenities**
- `Latitude`, `Longitude`, `zipcode_freq`
- Nearby: `bank`, `bus`, `hospital`, `mall`, `park`, `restaurant`, `school`, `station`, `supermarket`

**Economic Indicators**
- `Per_Capita_Income`, `Median_Rent`, `Median_Home_Value`
- `Median_Age`, `Unemployed_Population`

**Time Features**
- `year`, `month`, `quarter`

---

## 🛠️ Tech Stack

- Python 3.12
- pandas
- numpy
- scikit-learn
- matplotlib

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/dhriti09/house-price-predictor
cd house-price-predictor
```

2. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib
```

3. Open `main.ipynb` in VS Code or Jupyter and run all cells

---

## 📈 What the Model Does

1. Loads and explores the dataset
2. Cleans missing values
3. Splits data into training (80%) and testing (20%) sets
4. Trains a **Linear Regression** model
5. Evaluates using MSE and R² metrics
6. Visualizes Actual vs Predicted prices

---

## 🔮 Future Improvements

- Try Random Forest or XGBoost for better accuracy
- Add feature engineering and scaling
- Hyperparameter tuning
- Build a simple web interface for predictions

---

