# Google Play Store App Market Analysis

## Overview
This project analyzes a cleaned Google Play Store dataset to explore the Android app market landscape, covering category saturation, rating patterns, app size vs. installs, pricing and monetization trends, and a rating-based sentiment breakdown by category. The notebook combines static visualizations (`matplotlib`/`seaborn`) with interactive visualizations (`plotly`) to derive data-driven recommendations for a new app launch.

## Objective
The objective of this project is to explore key characteristics of the Google Play Store app ecosystem — including category distribution, user ratings, app size, pricing, and estimated revenue — in order to generate actionable insights for developers considering a new app launch.

## Dataset Description
- **Dataset Name:** `google_cleaned (2).csv`
- **Initial Shape:** 10,840 rows × 16 columns (reduced to 9,659 rows × 15 columns after cleaning)
- **Key Columns:**
  - `App` — name of the application
  - `Category` — app category (e.g., FAMILY, GAME, TOOLS)
  - `Rating` — average user rating
  - `Reviews` — number of user reviews
  - `Size` — app size
  - `Installs` — number of installs
  - `Type` — Free or Paid
  - `Price` — app price (for paid apps)
  - `Content Rating` — target audience/content rating
  - `Genres` — app genre(s)
  - `Current Ver`, `Android Ver` — version information
  - `Day`, `month`, `year` — date components (likely related to the app's last update date, though this was not explicitly labeled in the notebook)

Despite its filename suggesting the data had already been pre-cleaned (`google_cleaned`), the notebook found that several columns (`Installs`, `Price`, `Size`) still required additional cleaning, as detailed below. The dataset does not specify its original source beyond the local file path used to load it.

## Tools and Technologies Used
- **Language:** Python
- **Libraries:**
  - `pandas` — data manipulation and analysis
  - `numpy` — numerical computations
  - `matplotlib` — data visualization
  - `seaborn` — statistical data visualization
  - `plotly.express` — interactive visualizations
- **Environment:** Jupyter Notebook

## Project Workflow

### Data Collection
The dataset was loaded from a local CSV file (`google_cleaned (2).csv`) using `pandas.read_csv()`.

### Data Cleaning
- Dropped the unnamed index column (`Unnamed: 0`) carried over from the CSV file.
- Confirmed there were no missing values in the raw loaded columns (`isnull().sum()` returned 0 for all columns) prior to type-specific cleaning.
- Applied custom cleaning functions to convert string-formatted columns to proper numeric types:
  - `Installs`: removed commas and "+" symbols, then converted to integer.
  - `Price`: removed "$" symbols and converted to float (treating blank/"0" values as 0.0).
  - `Size`: converted values ending in "M" to kilobytes (multiplied by 1,000) and values ending in "k" to kilobytes directly; values such as "Varies with device" were converted to `NaN`, since they could not be parsed as a fixed size.
- Filled missing `Rating` values first using the median rating within each app's `Category`, then filled any remaining missing ratings with the overall median rating.
- Removed duplicate app entries by sorting by `Reviews` (descending) and keeping only the first (i.e., most-reviewed) occurrence of each `App` name.
- After cleaning, the dataset shape changed from 10,840 rows × 16 columns to 9,659 rows × 15 columns, with zero remaining missing values.

### Data Preprocessing
- Derived an estimated revenue metric for paid apps (`EstRevenue = Price × Installs`) to support the pricing/monetization analysis.
- Derived a `Sentiment` column by mapping the numeric `Rating` value into three categories: "Positive" (Rating ≥ 4.0), "Neutral" (3.0 ≤ Rating < 4.0), and "Negative" (Rating < 3.0). **Note:** despite being labeled "Sentiment Analysis on User Reviews" in the notebook, this is a simple rule-based classification derived directly from the existing `Rating` column, not a text-based sentiment analysis of actual review content. A commented-out line (`#!pip install textblob`) suggests that genuine text-based sentiment analysis using the TextBlob library may have been originally intended but was not carried out in the final notebook.

### Exploratory Data Analysis
- Analyzed the distribution of apps across categories to identify the most and least saturated categories.
- Analyzed the overall distribution of app ratings.
- Computed and visualized average rating by category.
- Explored the relationship between app size and number of installs, including a Pearson correlation calculation.
- Analyzed the proportion of Free vs. Paid apps and the price distribution among paid apps.
- Estimated revenue by category among paid apps.
- Derived and analyzed a rating-based sentiment classification, both overall and broken down by category.

### Visualization
A combination of static (`matplotlib`/`seaborn`) and interactive (`plotly`) visualizations were created (detailed in the **Visualizations Performed** section below).

### Modeling
No machine learning models were built in this notebook. The project is focused entirely on data cleaning and exploratory/descriptive market analysis.

### Evaluation
As no predictive model was built, there is no formal model evaluation (e.g., accuracy, RMSE) in this notebook. The closest equivalent is the before-vs-after data cleaning check (row count and missing value counts), which confirmed the cleaning steps were effective.

## Key Insights and Findings
- **Category saturation:** FAMILY (1,875 apps), GAME (945 apps), and TOOLS (829 apps) are by far the most saturated categories on the Play Store, followed by a "crowded middle" of BUSINESS, MEDICAL, PERSONALIZATION, PRODUCTIVITY, LIFESTYLE, FINANCE, and SPORTS (roughly 325–420 apps each). Categories such as Beauty, Comics, and Parenting have the lowest app counts, indicating less competition but also likely smaller target audiences.
- **Rating distribution:** App ratings are heavily left-skewed toward high values, with the distribution peaking sharply between 4.2 and 4.3; ratings below 3.0 are rare. This suggests a 4.0+ rating is a baseline market expectation rather than a differentiator.
- **Average rating by category:** Every category maintains an average rating above 4.0. EVENTS (4.363), ART_AND_DESIGN (4.354), and EDUCATION (4.353) have the highest average ratings, while categories like DATING, MAPS_AND_NAVIGATION, and TOOLS tend to sit closer to the 4.0 floor (based on the notebook's narrative description; the full printed ranking was truncated in the notebook output).
- **Size vs. installs:** The correlation between app `Size` and `Installs` is very weak (**0.0601**), indicating that app file size has little to no direct relationship with install volume. Apps of all sizes — from small to the maximum recorded size — can achieve very high install counts, though the majority of apps cluster at smaller file sizes.
- **Free vs. paid apps:** The vast majority of apps (92.2%) are free, with only 7.8% being paid — underscoring the dominance of freemium/ad-supported monetization models on the platform.
- **Paid app pricing:** Among the 754 paid apps, the average price is $14.06, but this is heavily influenced by high-priced outliers (max $400); the median price is much lower at $2.99, and 75% of paid apps are priced at $4.99 or below.
- **Estimated revenue by category:** FAMILY apps generate by far the highest estimated revenue among paid apps (~$115.7 million), followed by LIFESTYLE (~$57.6 million), GAME (~$40.7 million), and FINANCE (~$25.7 million) — indicating that paid-app monetization success is highly concentrated in a small number of categories.
- **Rating-based sentiment:** Using the Rating-derived sentiment classification, the large majority of apps fall into the "Positive" category (7,750 apps), with far fewer "Neutral" (1,631) and "Negative" (278) apps. Categories such as EVENTS (95.3% positive) and EDUCATION (94.4% positive) show the highest positive sentiment share.
- The notebook's final conclusion cell contains bracketed placeholder guidance (e.g., "[Fill in from your output — e.g. FAMILY and GAME categories are the most saturated...]") rather than a fully finalized write-up; the actual supporting figures needed to complete these insights are present elsewhere in the notebook's outputs and have been incorporated into the findings above.

## Visualizations Performed
1. **App Distribution Across Categories** — horizontal bar chart showing the number of apps per category.
2. **Distribution of App Ratings** — histogram (with KDE) of the `Rating` column.
3. **Average Rating by Category** — horizontal bar chart of mean rating per category.
4. **App Size vs. Installs** — scatter plot (log-scaled y-axis) showing the relationship between app size and installs.
5. **Free vs. Paid Apps** — pie chart showing the proportion of free vs. paid apps.
6. **Price Distribution — Paid Apps** — histogram of prices among paid apps.
7. **Estimated Revenue by Category (Paid Apps)** — horizontal bar chart of estimated total revenue per category.
8. **Sentiment Distribution by Category (%)** — stacked bar chart showing the percentage breakdown of Positive/Neutral/Negative rating-based sentiment per category.
9. **Interactive Scatter Plot (Plotly)** — Rating vs. Installs, with bubble size representing review count and color representing category.
10. **Interactive Bar Chart (Plotly)** — Average Rating by Category, with a continuous color scale.

## Results
- The Android app market is dominated by a small number of highly saturated categories (Family, Game, Tools), while niche categories offer lower competition but smaller audiences.
- User ratings across the platform are consistently high, with little differentiation between categories at the aggregate level — nearly all categories average above 4.0.
- App size is not a meaningful predictor of install success, suggesting that developer resources are better spent on other factors such as app store optimization and user experience.
- The free/freemium model overwhelmingly dominates the market, and paid app revenue is concentrated in a handful of categories (notably Family, Lifestyle, Game, and Finance).
- Rating-based sentiment is overwhelmingly positive across nearly all categories, though this reflects the existing rating distribution rather than an independent analysis of review text.

## Conclusion
Based on the notebook's exploratory findings, three data-driven insights are offered for a new app launch:

1. **Category saturation vs. opportunity** — FAMILY and GAME categories are the most saturated (highest app counts), meaning higher competition. Categories with fewer apps but solid average ratings may represent under-served niches worth targeting, based on a combined view of category counts and average ratings.
2. **Free-to-paid economics** — The vast majority of apps (~92%) are free, and paid apps cluster at low price points (median price of $2.99). A freemium model with in-app purchases is likely more aligned with market norms than an upfront paid model, unless targeting a category with historically higher paid-app tolerance and revenue potential, such as Family or Lifestyle.
3. **Size doesn't strongly predict installs** — The weak correlation (0.0601) between app size and installs suggests that app size is not a major driver of adoption. Investment is likely better directed toward App Store Optimization (ASO), category selection, and rating/review quality rather than minimizing app size alone.

## How to Run This Project
1. Clone or download this repository.
2. Ensure you have Python installed along with the following libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly jupyter
   ```
3. Place the dataset file `google_cleaned (2).csv` in the appropriate directory (update the file path in the notebook if necessary, as the original notebook references a local path).
4. Open the notebook using:
   ```bash
   jupyter notebook Level_no_2_Task_No_4.ipynb
   ```
5. Run all cells sequentially to reproduce the cleaning, EDA, and both static and interactive visualizations.

## Project Structure
```
├── Level_no_2_Task_No_4.ipynb     # Main Jupyter Notebook containing the full analysis
├── google_cleaned (2).csv         # Dataset used for analysis (not included; path must be set locally)
└── README.md                      # Project documentation
```

## Internship Details
- **Internship Provider:** Oasis Infobyte
- **Domain:** Data Analytics
- **Task Number:** Task 4 (Level 2)

## Author
Prepared as part of a Data Analytics internship task with Oasis Infobyte.
