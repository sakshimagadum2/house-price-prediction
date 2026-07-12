# 🏠 House Price Prediction

A machine learning project that classifies houses into **Low**, **Medium**, or **High** price categories based on property, location, and socio-economic features. Built for Machine Learning (24ECSC212), KLE Technological University.

## Overview

Instead of predicting an exact price, this project treats house price prediction as a **multi-class classification problem** — bucketing prices into three categories using `pd.qcut`. This is more interpretable and robust to price variance than regression.

## Dataset

- **Source:** [Kaggle House Price Dataset](https://www.kaggle.com/datasets/abhishek14398/house-price-prediction-dataset)
- **Records:** 50,000 | **Features:** 19 (17 numerical, 2 categorical: `location`, `income_level`)
- No missing values

**Feature groups:** property (area, bedrooms, bathrooms, floors, age), location & accessibility (distance, location, public_transport), amenities (garage, parking, garden, security), nearby facilities (school, hospital, mall), socio-economic (crime_rate, population_density, income_level), and `price` (used to derive the target).

## Pipeline

1. **Cleaning** — removed duplicates, fixed inconsistent category labels, dropped invalid/non-positive values, dropped remaining nulls
2. **Encoding** — mapped `location` (low/medium/premium) and `income_level` (low/mid/high) to numeric values
3. **Target creation** — `price` bucketed into `price_category`: **Low / Medium / High** via quantile binning (`q=3`)
4. **Scaling** — `StandardScaler` on numerical features
5. **Dimensionality reduction** — PCA retaining 95% variance
6. **Train-test split** — 80/20, stratified by `price_category`

## Models & Results

Four classifiers trained on the same PCA-reduced features:

| Model | Test Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **97.87%** | 97.87% | 97.87% | 97.87% |
| Naive Bayes | 95.09% | 95.08% | 95.09% | 95.08% |
| KNN (k=5) | 69.48% | 68.56% | 69.48% | 68.85% |
| Decision Tree | 66.17% | 66.40% | 66.17% | 66.28% |

**Best model: Logistic Regression** — highest test accuracy, with training and testing accuracy nearly equal (no overfitting). Decision Tree hit 100% training accuracy but only 66% on test data, a clear overfitting case. The trained model, scaler, and PCA transformer are saved together in `house_price_best_model.pkl` for reuse on new data.

## Key Findings

- Correlation analysis shows **area** has the strongest positive relationship with price, followed by **location** and **income_level**
- **Age**, **distance from city**, and **crime_rate** correlate negatively with price
- No strong correlation between independent features (no multicollinearity)

## Project Structure

```
house-price-prediction/
├── MLproject.ipynb              # EDA, preprocessing, modeling, evaluation
├── house_price.csv              # Dataset
├── house_price_best_model.pkl   # Saved model + scaler + PCA
├── README.md
├── .gitignore
└── LICENSE
```

## Getting Started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook MLproject.ipynb
```

## Team

| Name | USN |
|---|---|
| Sakshi Magadum | 02FE24BCS159 |
| Nikhita Patil | 02FE24BCS005 |
| Vidya Hugar | 02FE24BCS153 |
| Ritika Navalyal | 02FE24BCS151 |

Under the guidance of **Dr. Satish Bhojannawar**, Dept. of CSE, KLE Technological University

## License

This project is licensed under the [MIT License](LICENSE).
