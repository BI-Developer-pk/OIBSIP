# Dirty Café Sales — Data Cleaning and Exploratory Data Analysis

## Overview
This project focuses on cleaning a messy, real-world-style café sales dataset and then performing exploratory data analysis (EDA) on the cleaned data. The notebook demonstrates a full data cleaning pipeline — handling missing values, invalid entries, duplicates, and outliers — before analyzing sales patterns by item, location, payment method, and time.

## Objective
The objective of this project is to transform a "dirty" café sales dataset into a clean, analysis-ready dataset using documented data cleaning techniques, and to extract business insights related to product performance, sales channels, payment preferences, and monthly sales trends.

## Dataset Description
- **Dataset Name:** `dirty_cafe_sales.csv`
- **Initial Shape:** 10,000 rows
- **Initial Data Quality Issues:**
  - 102 missing/invalid entries (including values explicitly recorded as `"UNKNOWN"` and `"ERROR"`)
  - 0 exact duplicate rows detected initially (duplicates were later checked again based on `Transaction ID`)
- **Key Columns:**
  - `Transaction ID` — unique identifier for each transaction
  - `Item` — product purchased (e.g., Coffee, Cake, Cookie, Salad)
  - `Quantity` — number of units purchased
  - `Price Per Unit` — price per unit of the item
  - `Total Spent` — total amount spent on the transaction
  - `Payment Method` — method used for payment (e.g., Credit Card, Cash, Digital Wallet)
  - `Location` — where the purchase was made (In-store or Takeaway)
  - `Transaction Date` — date of the transaction

The dataset does not specify its original source; it was loaded directly from a local CSV file for analysis. The "dirty" nature of the dataset (containing placeholder values like `"ERROR"` and `"UNKNOWN"`) suggests it was intentionally designed for data cleaning practice.

## Tools and Technologies Used
- **Language:** Python
- **Libraries:**
  - `pandas` — data manipulation and cleaning
  - `numpy` — numerical operations
  - `matplotlib` — data visualization
  - `seaborn` — statistical data visualization
  - `os` — imported, though not directly used for file operations shown in the notebook
- **Environment:** Jupyter Notebook

## Project Workflow

### Data Collection
The dataset was loaded from a local CSV file (`dirty_cafe_sales.csv`) using `pandas.read_csv()`.

### Data Cleaning
The notebook follows a structured, multi-step cleaning process:

1. **Initial Data Quality Report:**
   - Recorded the initial row count (10,000).
   - Checked for exact duplicate rows (0 found initially).
   - Counted missing/invalid values by treating `"UNKNOWN"`, `"ERROR"`, empty strings, and whitespace as null (102 found).

2. **Standardization & Data Type Correction:**
   - Stripped whitespace and standardized text casing (capitalization) for `Transaction ID`, `Item`, `Payment Method`, and `Location`.
   - Replaced invalid placeholder values (`"Unknown"`, `"Error"`, `"Nan"`, empty/whitespace strings) with proper `NaN` values.
   - Converted `Transaction Date` to datetime format, coercing invalid entries to `NaN`.
   - Converted `Quantity`, `Price Per Unit`, and `Total Spent` to numeric types, coercing invalid entries to `NaN`.

3. **Missing Data Handling & Business Logic:**
   - **Price Per Unit:** Missing values were filled using the historical mode price for each specific item (since café prices are typically fixed per item); any remaining missing prices were filled with the overall median price.
   - **Quantity and Total Spent:** Missing `Quantity` values were filled with the column median, and `Total Spent` was recalculated using the relationship `Total Spent = Quantity × Price Per Unit`.
   - **Item, Payment Method, and Location:** Missing categorical values were filled using mode imputation (the most frequently occurring value in each column).
   - Any remaining missing values were handled with a backward-fill/forward-fill (`bfill().ffill()`).
   - Rows with missing `Transaction Date` were dropped entirely, since imputing dates was considered inappropriate for time-series integrity.

4. **Duplicate Removal:**
   - Duplicate transactions were removed based on `Transaction ID`, keeping the first occurrence.

5. **Outlier Detection and Treatment (IQR Method):**
   - The Interquartile Range (IQR) method was used on the `Quantity` column to identify outliers, chosen over the Z-score method for being more robust to skewed distributions.
   - Outliers above the upper bound (`Q3 + 1.5 × IQR`) were **capped** (not removed) to preserve legitimate high-volume purchases while limiting their statistical influence.
   - `Quantity` was then converted to integer type, and `Total Spent` was recalculated accordingly.

6. The cleaned dataset was exported to a new file: `cleaned_cafe_sales.csv`.

**Before vs. After Cleaning Summary (as reported in the notebook):**

| Metric | Before Cleaning | After Cleaning |
|---|---:|---:|
| Row Count | 10,000 | 9,540 |
| Total Null/Errors | 102 | 0 |
| Duplicate Count | 0 | 0 |

### Data Preprocessing
- Reloaded the cleaned dataset (`cleaned_cafe_sales.csv`) and re-parsed `Transaction Date` as a datetime column.
- Derived two new time-based features: `Year-Month` (year-month period) and `Month Name` (full month name), used for time series aggregation.

### Exploratory Data Analysis
- Analyzed total quantity sold and total revenue generated per item.
- Analyzed total revenue by sales location (In-store vs. Takeaway).
- Analyzed total revenue by payment method (Cash, Credit Card, Digital Wallet).
- Aggregated total revenue by month to observe sales trends over the course of 2023.

### Visualization
Charts were created using `matplotlib` and `seaborn`, including a bar chart, two pie/donut charts, and a line chart (detailed in the **Visualizations Performed** section below).

### Modeling
No machine learning models were built in this notebook. The project is focused entirely on data cleaning and exploratory/descriptive analysis.

### Evaluation
As no predictive model was built, there is no model evaluation (e.g., accuracy, RMSE) in this notebook. The "evaluation" component of this project is the before-vs-after data quality summary (row count, null/error count, and duplicate count), which was used to verify the effectiveness of the cleaning process.

## Key Insights and Findings
- The cleaning process reduced the dataset from 10,000 to 9,540 rows (a reduction of 460 rows, primarily due to dropped records with missing transaction dates), eliminated all 102 null/error values, and confirmed zero duplicate transactions in the final dataset.
- **Top-performing items by revenue:** Juice generated the highest total revenue ($18,249), followed by Salad ($16,570), Sandwich ($13,064), Smoothie ($12,832), and Cake ($9,834). Coffee, Tea, and Cookie generated comparatively lower total revenue ($6,850, $4,687.50, and $3,093 respectively), despite Coffee having a relatively high total quantity sold (3,425 units).
- **Sales by location:** Takeaway generated $59,296.50 in total revenue, compared to $25,883 for In-store. (Note: the notebook's narrative text describes this split as "almost equally" balanced between In-store (50.5%) and Takeaway (49.5%), but the underlying grouped values show Takeaway revenue is more than double In-store revenue. This discrepancy exists in the original notebook and is reported here as observed, rather than resolved with an assumption.)
- **Sales by payment method:** Digital Wallet was the leading payment method by revenue ($46,128), followed by Cash ($19,541) and Credit Card ($19,510.50).
- **Monthly sales trend (2023):** Monthly revenue ranged narrowly between approximately $6,640 (February) and $7,376.50 (June), with no extreme seasonal spikes or drops in the actual grouped data. (Note: the notebook's narrative text describes a "summer slump" bottoming near $13,000 and an "autumn comeback" peaking near $16,500; these specific figures do not match the actual monthly totals computed in the notebook, which range from about $6,640 to $7,377. This discrepancy exists in the original notebook and is reported here as observed, rather than resolved with an assumption.)

## Visualizations Performed
1. **Total Revenue by Item** — bar chart showing revenue generated by each menu item.
2. **Sales Share by Location** — donut/pie chart showing the proportion of revenue from In-store vs. Takeaway sales.
3. **Sales Share by Payment Method** — donut/pie chart showing the proportion of revenue by payment method (Cash, Credit Card, Digital Wallet).
4. **Monthly Sales Trend Over Time** — line chart showing total monthly revenue across 2023.

## Results
- The data cleaning pipeline successfully removed all identified missing values, invalid placeholder entries, and duplicate transactions, resulting in a clean dataset of 9,540 verified records.
- Juice, Salad, and Sandwich are the top revenue-generating menu items, while Cookie and Tea are the lowest.
- Takeaway is the larger revenue channel compared to In-store dining based on the grouped totals.
- Digital Wallet is the most-used payment method by revenue, indicating a strong preference for cashless transactions among customers.
- Monthly revenue throughout 2023 remained relatively stable, without the large seasonal swings described in the notebook's narrative commentary.

## Conclusion
This project demonstrates a complete, well-documented data cleaning workflow — including missing value imputation based on business logic, categorical mode imputation, robust outlier handling via the IQR method, and duplicate removal — applied to a deliberately "dirty" café sales dataset. The resulting clean dataset enabled meaningful exploratory analysis of product performance, sales channels, and payment preferences. Based on the findings, the notebook proposes the following strategic recommendations for the café:

1. **Boost Takeaway Speed** — Since takeaway represents a large share of total sales, introducing a dedicated "Express Pickup Lane" or an online pre-ordering system could improve customer satisfaction.
2. **Cashless Incentives** — Introduce small digital loyalty rewards (e.g., a free item after a set number of digital wallet payments) to encourage faster, smoother checkouts, given the strong preference for digital wallets.
3. **Menu Bundling** — Pair high-volume, lower-priced drinks (e.g., Coffee, Tea) with high-margin food items (e.g., Sandwiches, Cakes) as combo deals to increase the Average Order Value (AOV).

## How to Run This Project
1. Clone or download this repository.
2. Ensure you have Python installed along with the following libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```
3. Place the dataset file `dirty_cafe_sales.csv` in the appropriate directory (update the file path in the notebook if necessary, as the original notebook references a local path).
4. Open the notebook using:
   ```bash
   jupyter notebook Level_no_1_Task_No_3.ipynb
   ```
5. Run all cells sequentially. The notebook will generate a cleaned dataset file (`cleaned_cafe_sales.csv`) and reproduce all EDA visualizations.

## Project Structure
```
├── Level_no_1_Task_No_3.ipynb     # Main Jupyter Notebook containing the cleaning pipeline and EDA
├── dirty_cafe_sales.csv           # Raw/original dataset (not included; path must be set locally)
├── cleaned_cafe_sales.csv         # Cleaned dataset generated by the notebook
└── README.md                      # Project documentation
```

## Internship Details
- **Internship Provider:** Oasis Infobyte
- **Domain:** Data Analytics
- **Task Number:** Task 3 (Level 1)

## Author
Prepared as part of a Data Analytics internship task with Oasis Infobyte.
