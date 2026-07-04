# Sales and Customer Data Analysis

## Overview
This project is a data analytics case study performed on a sales and customer dataset. The notebook covers data inspection, descriptive statistics, time series analysis, customer demographics analysis, product category and revenue analysis, correlation analysis, retention strategy evaluation, and customer segmentation, concluding with business recommendations derived from the findings.

## Objective
The objective of this project is to analyze customer purchasing behavior, sales trends, and retention strategies in order to uncover actionable insights that can help a business improve revenue, customer retention, and marketing effectiveness.

## Dataset Description
- **Dataset Name:** `Sales_and_Customer_data.csv`
- **Shape:** 10,000 rows × 15 columns
- **Missing Values:** None (all columns have 0 null values)
- **Key Columns:**
  - `Customer_ID`, `Product_ID`, `Transaction_ID` — unique identifiers
  - `Purchase_Frequency` — number of purchases made by a customer
  - `Average_Order_Value` — average value of an order
  - `Most_Frequent_Category` — most purchased product category
  - `Time_Between_Purchases` — days between purchases
  - `Region` — customer's geographic region
  - `Churn_Probability` — probability of customer churn
  - `Lifetime_Value` — customer lifetime value (LTV)
  - `Launch_Date`, `Peak_Sales_Date` — relevant product/sales dates
  - `Season` — seasonal tag associated with the transaction
  - `Preferred_Purchase_Times` — preferred time of day for shopping (e.g., Morning, Afternoon, Evening)
  - `Retention_Strategy` — retention strategy applied to the customer (e.g., Loyalty Program, Discounts, Email Campaigns)

The dataset does not specify its original source; it was loaded directly from a local CSV file for analysis.

## Tools and Technologies Used
- **Language:** Python
- **Libraries:**
  - `pandas` — data manipulation and analysis
  - `numpy` — numerical computations
  - `matplotlib` — data visualization
  - `seaborn` — statistical data visualization
- **Environment:** Jupyter Notebook

## Project Workflow

### Data Collection
The dataset was loaded from a local CSV file (`Sales_and_Customer_data.csv`) using `pandas.read_csv()`.

### Data Cleaning
- Checked dataset shape (10,000 rows, 15 columns).
- Reviewed column data types using `df.info()`.
- Checked for missing values using `df.isnull().sum()` — the dataset was found to be completely clean with zero missing values.
- No duplicate handling, outlier treatment, or data type corrections were documented in the notebook beyond the missing value check.

### Data Preprocessing
- Converted the `Peak_Sales_Date` column to a proper datetime format using `pd.to_datetime()`.
- Derived new time-based features: `Year_Month` and `Quarter`, extracted from `Peak_Sales_Date` for time series aggregation.
- Created a jittered version of `Purchase_Frequency` (`Purchase_Frequency_Jittered`) to reduce overplotting in the customer segmentation scatter plot.

### Exploratory Data Analysis
- Generated descriptive statistics (mean, std, min, max, quartiles) for numerical columns using `df.describe()`.
- Computed the mode of numerical columns to understand the most frequently occurring values.
- Analyzed monthly sales trends by grouping `Average_Order_Value` by `Year_Month`.
- Examined customer distribution across regions and preferred shopping times.
- Analyzed total revenue contribution by product category.
- Built a correlation matrix of numerical variables.
- Compared churn probability across different retention strategies.
- Performed a cross-analysis between product category and preferred purchase time.
- Segmented customers based on purchase frequency and lifetime value.

### Visualization
Multiple charts were created using `matplotlib` and `seaborn`, including line charts, count plots, bar plots, heatmaps, box plots, and scatter plots (detailed in the **Visualizations Performed** section below).

### Modeling
No machine learning models were built in this notebook. The analysis is entirely descriptive/exploratory in nature, based on statistical summaries and visual analytics.

### Evaluation
As no predictive model was built, there is no model evaluation (e.g., accuracy, RMSE) in this notebook. Evaluation in this project instead refers to the interpretation of visual and statistical patterns to derive business insights.

## Key Insights and Findings
- The dataset is fully clean, with 10,000 records and no missing values across any of the 15 columns.
- The average purchase frequency is approximately 10, the average order value is close to $110, and the average Customer Lifetime Value (LTV) is approximately $5,031.93.
- Monthly sales show high volatility, with peak sales recorded in January and May (both crossing 96,000) and the lowest point in February (around 85,000). Sales recover to approximately $95,000 by December.
- Customer distribution is fairly even across regions, with Europe slightly leading at 2,547 customers. Evening is the most preferred shopping time, with 3,433 customers.
- Revenue contribution across product categories is well balanced: Electronics leads at $282,392, closely followed by Clothing and Home, while Sports is the lowest performer at $266,356.
- Correlation analysis indicates that customer engagement metrics are stronger drivers of revenue and lifetime value than demographic variables, suggesting that improving retention and purchase frequency could have the greatest business impact.
- All three retention strategies (Loyalty Program, Discounts, Email Campaigns) show similar median churn probabilities (around 0.5), with the Loyalty Program performing marginally better.
- Shopping behavior by time of day is fairly consistent: Evening is the peak window for Clothing and Home (35.0% of category orders each), Afternoon is preferred for Sports (34.1%), and Electronics sales are nearly flat between Afternoon and Evening. Morning is consistently the slowest shopping window across categories.
- Customer segmentation based on Purchase Frequency vs. Lifetime Value shows a fairly uniform distribution across regions, with no distinct clustering, and forms four evenly populated quadrants around the average LTV ($5,000) and average frequency (10) — indicating that frequency and LTV behave as independent metrics in this dataset.

## Visualizations Performed
1. **Monthly Total Sales Volume Trend** — line chart showing sales fluctuation across months.
2. **Customer Distribution by Region** — count plot of customers per region.
3. **Preferred Shopping Time Distribution** — count plot of customers by preferred purchase time.
4. **Total Revenue Contribution by Product Category** — bar chart of revenue by category.
5. **Correlation Matrix Heatmap** — heatmap of correlations among numerical variables.
6. **Effect of Retention Strategies on Customer Churn Risk** — box plot of churn probability by retention strategy.
7. **Category vs. Preferred Shopping Time Heatmap** — cross-tabulation heatmap showing purchase timing behavior per category.
8. **Customer Strategic Segmentation (LTV vs. Purchase Frequency)** — scatter plot segmenting customers by region, with average LTV and frequency reference lines.

## Results
- The business shows a diversified and balanced revenue base across product categories, with no single category dominating sales.
- Customer engagement (frequency, retention) is more influential on revenue and LTV than demographic factors like region.
- Retention strategies currently in use produce similar churn outcomes, suggesting room for more targeted, data-driven retention approaches.
- Customer purchase timing patterns are moderately consistent across product categories, offering opportunities for time-targeted marketing.

## Conclusion
The analysis shows that the business has a stable, well-diversified customer base and revenue stream, with purchasing behavior that varies moderately by category and time of day but not strongly by region. Customer engagement metrics (purchase frequency and retention) appear to be more impactful on revenue and lifetime value than demographic attributes. Based on these findings, the notebook proposes the following business recommendations:

1. **Hyper-Personalized Time-Based Marketing** — Align promotional notifications with category-specific peak shopping windows (e.g., Evening or Afternoon), scheduling automated SMS/email campaigns roughly one hour prior to a customer's peak historic activity window.
2. **Geographic Distribution of Inventory** — Calibrate localized supply chains and fulfillment hubs based on regional category preferences to reduce logistics overhead and shipping transit times.
3. **Data-Driven Retention Rules** — Continually audit retention strategy outcomes, prioritizing strategies with the lowest churn probability or highest correlated LTV, while phasing out low-impact campaigns.

## How to Run This Project
1. Clone or download this repository.
2. Ensure you have Python installed along with the following libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```
3. Place the dataset file `Sales_and_Customer_data.csv` in the appropriate directory (update the file path in the notebook if necessary, as the original notebook references a local path).
4. Open the notebook using:
   ```bash
   jupyter notebook Level_no_1_Task_No_1.ipynb
   ```
5. Run all cells sequentially to reproduce the analysis and visualizations.

## Project Structure
```
├── Level_no_1_Task_No_1.ipynb     # Main Jupyter Notebook containing the full analysis
├── Sales_and_Customer_data.csv    # Dataset used for analysis (not included; path must be set locally)
├── monthly_sales_trend.png        # Generated visualization
├── customer_demographics_analysis.png  # Generated visualization
├── product_category_revenue.png   # Generated visualization
├── correlation_matrix.png         # Generated visualization
├── retention_vs_churn.png         # Generated visualization
├── category_vs_time_heatmap.png   # Generated visualization
├── vip_segmentation_clean.png     # Generated visualization
└── README.md                      # Project documentation
```

## Internship Details
- **Internship Provider:** Oasis Infobyte
- **Domain:** Data Analytics
- **Task Number:** Task 1 (Level 1)

## Author
Prepared as part of a Data Analytics internship task with Oasis Infobyte.
