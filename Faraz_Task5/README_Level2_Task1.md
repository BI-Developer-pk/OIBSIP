# House Price Prediction — Regression Analysis

## Overview
This project analyzes a house price dataset and builds regression models to predict property prices based on structural, categorical, and locational features. The notebook covers exploratory data analysis, feature importance evaluation using statistical tests and a Random Forest model, categorical encoding, and the training and evaluation of Linear Regression, Ridge, and Lasso regression models.

## Objective
The objective of this project is to explore the relationship between house features (such as area, bedrooms, condition, and location) and sale price, identify which features most strongly influence price, and build and evaluate regression models capable of predicting house prices.

## Dataset Description
- **Dataset Name:** `House Price Prediction Dataset.csv`
- **Shape:** 2,000 rows × 10 columns
- **Missing Values:** None (confirmed via `df.isnull().sum()`)
- **Key Columns:**
  - `Id` — record identifier
  - `Area` — house area (square footage)
  - `Bedrooms` — number of bedrooms
  - `Bathrooms` — number of bathrooms
  - `Floors` — number of floors
  - `YearBuilt` — year the house was built
  - `Location` — categorical location type (Downtown, Suburban, Rural, Urban)
  - `Condition` — categorical property condition (Excellent, Good, Fair, Poor)
  - `Garage` — whether the property has a garage (Yes/No)
  - `Price` — target variable; the sale price of the house

The dataset does not specify its original source; it was loaded directly from a local CSV file for analysis.

## Tools and Technologies Used
- **Language:** Python
- **Libraries:**
  - `pandas` — data manipulation and analysis
  - `numpy` — numerical computations
  - `matplotlib` — data visualization
  - `seaborn` — statistical data visualization
  - `scipy` (`stats.f_oneway`) — one-way ANOVA testing for categorical feature significance
  - `scikit-learn` — feature importance (`RandomForestRegressor`), preprocessing (`LabelEncoder`, `pd.get_dummies`), model training and evaluation (`train_test_split`, `LinearRegression`, `Ridge`, `Lasso`, `mean_squared_error`, `r2_score`)
- **Environment:** Jupyter Notebook

## Project Workflow

### Data Collection
The dataset was loaded from a local CSV file (`House Price Prediction Dataset.csv`) using `pandas.read_csv()`.

### Data Cleaning
- Verified there were no missing values across any of the 10 columns using `df.isnull().sum()`.
- Reviewed column data types using `df.dtypes`, confirming numerical columns (`Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`, `Price`) and categorical/object columns (`Location`, `Condition`, `Garage`).
- As an added safeguard, the notebook also included a step to fill missing numeric values with the column median and missing categorical values with the column mode, even though no missing values were actually present in this dataset.

### Data Preprocessing
- Categorical columns (`Location`, `Condition`, `Garage`) were encoded in two ways at different stages of the notebook:
  - `LabelEncoder` was used temporarily to encode categorical features for the Random Forest feature importance step.
  - `pd.get_dummies()` (one-hot encoding, with `drop_first=True`) was used to encode categorical features for the regression modeling pipeline.
- The `Id` column was considered for removal (a commented-out line `#df = df.drop(columns=['Id'])` shows this was discussed but not actually executed in the final notebook).
- Data was split into training and testing sets using an 80/20 split (`test_size=0.2`, `random_state=42`).

### Exploratory Data Analysis
- Reviewed dataset shape, structure, data types, and summary statistics (`df.describe(include='all')`).
- Analyzed the distribution and skewness of the `Price` variable (skewness calculated as **-0.064**, indicating a nearly symmetric distribution).
- Computed Pearson correlations between numeric features and `Price`.
- Conducted one-way ANOVA (`f_oneway`) tests to assess whether `Location`, `Condition`, and `Garage` had statistically significant effects on mean `Price`.
- Computed Random Forest feature importances to rank the relative influence of all features on `Price`.
- Generated a full correlation heatmap of the one-hot-encoded dataset and reviewed the correlation of each feature with `Price`.

### Visualization
Numerous charts were created using `matplotlib` and `seaborn` (detailed in the **Visualizations Performed** section below).

### Modeling
Four regression models were trained and compared to predict `Price`:

1. **Linear Regression**
2. **Ridge Regression** (`alpha=1.0`)
3. **Lasso Regression** (`alpha=1.0`, `max_iter=10000`)
4. A **Random Forest Regressor** (`n_estimators=200`, `random_state=42`) was also used, but only for feature importance ranking, not as a final predictive model for comparison against the linear models.

All linear models were trained on the same 80/20 train/test split of the one-hot-encoded feature set.

### Evaluation
Models were evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score:

| Model | MSE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 78,321,466,146.03 | 279,859.73 | -0.0067 |
| Ridge (alpha=1.0) | 78,321,270,000 (approx.) | 279,859.37 | -0.0067 |
| Lasso (alpha=1.0) | 78,321,370,000 (approx.) | 279,859.55 | -0.0067 |

All three models produced nearly identical, very poor results, with **negative R² scores**, meaning each model performed slightly worse than simply predicting the mean price for every observation. A residual plot and an "Actual vs. Predicted" scatter plot were also generated to visually assess model fit; both indicate that predictions do not track actual prices in any meaningful pattern.

The notebook additionally checked whether Lasso regularization had zeroed out any feature coefficients (a sign of feature selection); the check confirmed no coefficients were reduced to zero at `alpha=1.0`.

## Key Insights and Findings
- **The dataset shows essentially no meaningful linear relationship between any feature and `Price`.** Correlations between numeric features and `Price` were all extremely close to zero (Floors: 0.056, Bathrooms: -0.016, YearBuilt: 0.005, Bedrooms: -0.003, Area: 0.0015).
- ANOVA tests on the categorical features (`Location`, `Condition`, `Garage`) all returned **p-values well above 0.05** (0.1943, 0.1826, and 0.8989 respectively), indicating none of these categorical features had a statistically significant effect on mean price.
- Random Forest feature importances ranked `Area` (0.335) and `YearBuilt` (0.268) as the most influential features, but this ranking reflects relative importance within a weak model rather than genuine strong predictive power, consistent with the near-zero correlations found elsewhere.
- The final regression models (Linear Regression, Ridge, and Lasso) all produced a **negative R² score (approximately -0.0067)**, meaning none of them explained any variance in house prices — they performed worse than a naive baseline that always predicts the mean price.
- **Important discrepancy:** several markdown "data story" cells in the notebook describe strong relationships that are not supported by the actual computed statistics — for example, one cell states that `Area` has "a powerful correlation coefficient of 0.81" with `Price` and describes a "clear, tight upward trajectory" in a scatter plot of Area vs. Price. However, the notebook's own computed correlation value for `Area` vs. `Price` is only **0.0015**, and the model evaluation results confirm no predictive relationship exists. This inconsistency exists in the original notebook and is reported here as observed, rather than resolved with an assumption. Overall, this suggests the dataset used may contain randomly generated or synthetic price values that are not causally linked to the listed house features.
- A code error/inconsistency was also observed: cell 19 (which plots Random Forest feature importance next to "Mutual Information Scores") references a variable `mi_df` that is not defined anywhere else in the notebook. Mutual information scores do not appear to have been computed in any visible code cell, though the notebook's saved output shows a chart was rendered — this suggests the `mi_df` variable may have been defined in a prior, undocumented run that was not included in the final saved notebook.

## Visualizations Performed
1. **Distribution of Price (Histogram with KDE) and Price Boxplot** — showing the overall spread and shape of the price variable.
2. **Random Forest Feature Importance and Mutual Information Scores (side-by-side bar charts)** — ranking feature influence on price (note: the Mutual Information portion relies on an undefined variable, as described above).
3. **Correlation Heatmap** — correlation matrix of all one-hot-encoded numeric and categorical features against `Price`.
4. **Actual vs. Predicted House Prices (Scatter Plot)** — comparing Linear Regression predictions to actual test-set prices, with a reference "perfect prediction" line.
5. **Residuals vs. Predicted Plot and Distribution of Residuals (Histogram)** — assessing model error patterns.
6. **Linear Regression Coefficients (Horizontal Bar Chart)** — visualizing the magnitude and direction of each feature's coefficient.
7. **Model Comparison: R² Score (Bar Chart)** — comparing R² scores across Linear Regression, Ridge, and Lasso models.

Additionally, the markdown "data story" cells throughout the notebook reference several boxplots (Price by Location, Price by Condition, Price by Garage, Price by Floors) and an Area vs. Price scatter plot; however, the actual plotting code for these specific charts was not present among the code cells that were reviewed, so their existence in the final rendered notebook could not be independently confirmed from the code alone.

## Results
- None of the three regression models tested (Linear Regression, Ridge, Lasso) were able to produce a usable predictive model for house price, all returning a negative R² score.
- Statistical testing (correlation analysis and ANOVA) consistently showed that none of the available features — including Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, or Garage — have a statistically meaningful relationship with Price in this dataset.
- The Random Forest-based feature importance ranking placed Area and YearBuilt at the top, but this should be interpreted cautiously given the overall lack of predictive signal in the dataset.

## Conclusion
Based on the statistical tests and model evaluation results in this notebook, the house price data used does not exhibit a meaningful relationship between the available house features and price. Correlation values are near zero across the board, ANOVA tests show no statistically significant categorical effects, and all three regression models (Linear, Ridge, and Lasso) achieved a negative R² score, indicating they performed worse than a simple mean-based baseline prediction. While several markdown commentary cells in the notebook describe strong, intuitive relationships (e.g., "Area is the primary driver of property value" with a stated correlation of 0.81), these narrative claims are not supported by the notebook's own computed statistics, which show correlations near zero and negative R² across all models. This suggests the underlying dataset may be synthetic or randomly generated with respect to its price values, and that any real-world housing price intuition described in the commentary should not be taken as a validated finding of this analysis. A more reliable conclusion, grounded strictly in the notebook's computed outputs, is that **none of the tested linear models were able to explain house price variation using the available features**.

## How to Run This Project
1. Clone or download this repository.
2. Ensure you have Python installed along with the following libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
   ```
3. Place the dataset file `House Price Prediction Dataset.csv` in the appropriate directory (update the file path in the notebook if necessary, as the original notebook references a local path).
4. Open the notebook using:
   ```bash
   jupyter notebook Level_no_2_Task_No_1.ipynb
   ```
5. Run all cells sequentially. Note: as discussed above, the cell referencing `mi_df` (Mutual Information Scores) may raise a `NameError` if run from a fresh kernel, since this variable was not defined in any preceding cell in the notebook as reviewed.

## Project Structure
```
├── Level_no_2_Task_No_1.ipynb          # Main Jupyter Notebook containing the full analysis
├── House Price Prediction Dataset.csv  # Dataset used for analysis (not included; path must be set locally)
└── README.md                           # Project documentation
```

## Internship Details
- **Internship Provider:** Oasis Infobyte
- **Domain:** Data Analytics
- **Task Number:** Task 1 (Level 2)

## Author
Prepared as part of a Data Analytics internship task with Oasis Infobyte.
