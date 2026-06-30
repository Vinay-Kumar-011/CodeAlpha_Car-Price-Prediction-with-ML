# 🚗 Car Price Prediction with Machine Learning

A regression project that predicts used-car selling prices from listing attributes such as present (showroom) price, mileage, fuel type, transmission, and car age.

## 📊 Dataset

The dataset (`car data.csv`) contains 301 used-car listings with the following fields:

| Column | Description |
|---|---|
| Car_Name | Model name of the car |
| Year | Year of manufacture |
| Selling_Price | Price the car was sold for (target variable, in lakhs) |
| Present_Price | Current showroom price of the car (in lakhs) |
| Driven_kms | Total kilometers driven |
| Fuel_Type | Petrol / Diesel / CNG |
| Selling_type | Dealer or Individual seller |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

## 🔍 Project Workflow

1. **Data Loading & Inspection** — shape, dtypes, missing values, and duplicate checks.
2. **Exploratory Data Analysis (EDA)**
   - Value counts and count plots for categorical columns
   - Histograms + KDE plots and boxplots for numeric columns (outlier inspection)
   - Scatter plots of price against mileage and car age
   - Correlation heatmap across numeric features
3. **Feature Engineering**
   - Log-transformed `Selling_Price` and `Present_Price` to reduce right-skew
   - Capped `Driven_kms` at the 95th percentile to limit the influence of extreme outliers
   - Derived `Age` of the car from manufacture year, then dropped the raw `Year` column
4. **Data Transformation**
   - Label-encoded categorical features (`Car_Name`, `Fuel_Type`, `Selling_type`, `Transmission`)
   - Standardized numeric features with `StandardScaler`
5. **Train/Test Split** — 80/20 split with a fixed random seed for reproducibility
6. **Modeling** — three regressors trained and evaluated with 5-fold cross-validation:

   | Model | RMSE (CV) | R² (CV) |
   |---|---|---|
   | **Linear Regression** | **0.151** | **0.962** |
   | SVM (SVR) | 0.175 | 0.953 |
   | Decision Tree | 0.231 | 0.910 |

7. **Model Comparison** — bar chart comparing RMSE and R² across all three models.

## 🏆 Key Results

- **Linear Regression** was the best-performing model on this dataset, explaining ~96% of the variance in (log) selling price with the lowest cross-validated RMSE.
- `Present_Price` is the strongest predictor of resale value, confirming that showroom price anchors a car's market value.
- Car age and kilometers driven both negatively correlate with selling price, consistent with real-world depreciation patterns.

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy** — data handling
- **Matplotlib**, **Seaborn** — visualization
- **Scikit-learn** — preprocessing, modeling (Linear Regression, SVR, Decision Tree), cross-validation

## 📁 Repository Structure

```
.
├── car-analysis.ipynb     # Full notebook: EDA → feature engineering → modeling
├── car data.csv            # Dataset
└── README.md
```

## ▶️ How to Run

1. Clone this repository and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Open `car-analysis.ipynb` in Jupyter or Kaggle and run all cells in order.

## 📌 Future Improvements

- Try ensemble models (Random Forest, Gradient Boosting/XGBoost) for potential accuracy gains
- Perform hyperparameter tuning (GridSearchCV) on SVR and Decision Tree
- Incorporate brand-level features for finer-grained "brand goodwill" effects
- Deploy the trained model as a simple web app (Streamlit/Flask) for interactive price predictions

## 📄 License

This project is open-sourced for educational purposes.
