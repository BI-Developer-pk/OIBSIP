# Red Wine Quality Classification

## Overview
This project analyzes the physicochemical properties of red wine samples and builds machine learning classification models to predict whether a wine can be considered "good quality." The notebook covers exploratory data analysis, a detailed class imbalance discussion, feature engineering through quality-score binning, and the training and comparison of three classification models using stratified splitting, scaling, and ROC/AUC evaluation.

## Objective
The objective of this project is to explore the relationship between a red wine's chemical properties and its quality rating, address the significant class imbalance present in the raw quality scores, and build and compare classification models capable of distinguishing "good" wines from the rest.

## Dataset Description
- **Dataset Name:** `winequality-red.csv`
- **Shape:** 1,599 rows × 12 columns
- **Missing Values:** None (confirmed via `df.isnull().sum()`)
- **Key Columns:**
  - `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol` — numeric physicochemical properties of each wine sample
  - `quality` — the target variable; an integer quality score ranging from 3 to 8, as assessed by wine tasters
- **Original Quality Score Distribution:**
  | Quality Score | Count |
  |---|---|
  | 3 | 10 |
  | 4 | 53 |
  | 5 | 681 |
  | 6 | 638 |
  | 7 | 199 |
  | 8 | 18 |

The dataset does not specify its original source; it was loaded directly from a local CSV file for analysis. Its structure and column names are consistent with the well-known UCI/Wine Quality dataset format, though the notebook itself does not explicitly cite this origin.

## Tools and Technologies Used
- **Language:** Python
- **Libraries:**
  - `pandas` — data manipulation and analysis
  - `numpy` — numerical computations
  - `matplotlib` — data visualization
  - `seaborn` — statistical data visualization
  - `scikit-learn` — preprocessing (`StandardScaler`), splitting (`train_test_split`), modeling (`RandomForestClassifier`, `SGDClassifier`, `SVC`), and evaluation (`accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`)
- **Environment:** Jupyter Notebook

## Project Workflow

### Data Collection
The dataset was loaded from a local CSV file (`winequality-red.csv`) using `pandas.read_csv()`.

### Data Cleaning
- Verified there were no missing values across any of the 12 columns using `df.isnull().sum()`.
- Reviewed dataset structure and data types using `df.info()`, confirming all columns are numeric (`float64`), with `quality` as an integer-scaled score.
- No duplicate handling or outlier treatment steps were documented in the notebook.

### Data Preprocessing
- **Class imbalance analysis:** Calculated the class imbalance ratio between the most and least frequent quality scores, finding a ratio of **68.1** (i.e., quality score 5 has 68.1 times more samples than quality score 3).
- **Feature engineering via binning:**
  - Created a binary target, `quality_binary`, defined as 1 if `quality >= 7` ("good") and 0 otherwise ("not good"), resulting in 217 "good" wines vs. 1,382 "not good" wines.
  - Also created an alternative 3-class target, `quality_3class`, binning scores into `low` (≤4), `medium` (5–6), and `high` (≥7) categories, resulting in 63 "low", 1,319 "medium", and 217 "high" samples.
  - The notebook explicitly chose to proceed with the **binary framing** as the primary modeling target, reasoning that it aligns with AUC/ROC-based evaluation, produces a more usable minority class than the near-empty extreme scores (3 and 8), and is more business-relevant (e.g., "is this wine good enough to sell as premium").
- **Train/test split:** Split the data using a 75/25 split (`test_size=0.25`, `random_state=42`), stratified by the `quality_binary` target to preserve class ratios in both sets (confirmed: ~86.4%/13.6% split preserved in both train and test sets).
- **Feature scaling:** Applied `StandardScaler` to the features, producing a scaled version of the training and test sets for use with the SGD and SVC classifiers (Random Forest was trained on the unscaled features, as tree-based models do not require feature scaling).

### Exploratory Data Analysis
- Reviewed the distribution of wine quality scores and confirmed the strong skew toward the middle scores (5 and 6).
- Plotted histograms (with KDE) for all 11 numeric physicochemical features to inspect their individual distributions.
- Generated a full correlation heatmap of all numeric features and examined the correlation of each feature with `quality`.
- Conducted a written class imbalance discussion identifying the risks of training a classifier on the heavily imbalanced raw quality scores.

### Visualization
Multiple charts were created using `matplotlib` and `seaborn` (detailed in the **Visualizations Performed** section below).

### Modeling
Three classification models were trained to predict the binary `quality_binary` target, all using `class_weight='balanced'` to help address class imbalance:

1. **Random Forest Classifier** (`n_estimators=200`, `random_state=42`, `class_weight='balanced'`) — trained on unscaled features.
2. **SGD Classifier** (`loss='log_loss'`, `random_state=42`, `class_weight='balanced'`) — trained on scaled features.
3. **Support Vector Classifier (SVC)** (`probability=True`, `random_state=42`, `class_weight='balanced'`) — trained on scaled features.

### Evaluation
Each model was evaluated on the held-out test set using accuracy, a full classification report (precision, recall, f1-score), a confusion matrix, and ROC/AUC analysis:

| Model | Accuracy | AUC | Precision (Good) | Recall (Good) | F1 (Good) |
|---|---|---|---|---|---|
| Random Forest | 0.9400 | 0.9466 | 0.9167 | 0.6111 | 0.7333 |
| SVC | 0.8325 | 0.8958 | 0.4316 | 0.7593 | 0.5503 |
| SGD | 0.7200 | 0.8342 | 0.2958 | 0.7778 | 0.4286 |

Confusion matrices and ROC curves (with AUC scores) were plotted for all three models to visually compare their performance, especially on the minority "good wine" class. Random Forest feature importances were also computed and visualized to identify which chemical properties most strongly influenced the model's predictions.

## Key Insights and Findings
- Wine quality scores in the raw dataset are heavily imbalanced: scores of 5 and 6 dominate the dataset (681 and 638 samples respectively out of 1,599), while extreme scores (3, 4, and 8) are rare (10, 53, and 18 samples respectively), producing a class imbalance ratio of 68.1 between the most and least common scores.
- Binning the quality scores into a binary "good" (≥7) vs. "not good" (<7) target created a more usable, though still imbalanced, classification problem (about 13.6% positive class).
- Correlation analysis showed that `alcohol` (0.476) has the strongest positive correlation with quality, followed by `sulphates` (0.251) and `citric acid` (0.226). `Volatile acidity` (-0.391) has the strongest negative correlation with quality.
- Random Forest feature importance rankings were broadly consistent with the correlation findings, ranking `alcohol` (0.252) as the most important feature, followed by `sulphates` (0.127) and `volatile acidity` (0.101).
- **Random Forest was the best-performing model overall**, achieving the highest accuracy (94.0%) and the highest AUC (0.9466), along with the best precision (0.9167) and F1-score (0.7333) on the "good" wine class, though its recall on the "good" class (0.6111) was lower than that of SVC and SGD.
- **SGD and SVC achieved higher recall on the minority "good" class** (0.7778 and 0.7593 respectively) than Random Forest, but at the cost of substantially lower precision (0.2958 and 0.4316 respectively), meaning they identified more of the actual "good" wines but also produced many more false positives.
- The notebook's own conclusion cell contains a placeholder instruction ("[fill in after running — likely Random Forest]") that was not completed with a final selection in the notebook text itself; however, based on the actual evaluation metrics reported above, Random Forest is the strongest performer by both accuracy and AUC.

## Visualizations Performed
1. **Distribution of Wine Quality Scores** — count plot of the raw `quality` variable (scores 3–8).
2. **Feature Distribution Histograms** — a grid of histograms (with KDE overlays) for all 11 numeric physicochemical features.
3. **Correlation Heatmap** — correlation matrix of all numeric features, including `quality`.
4. **Confusion Matrices** — side-by-side confusion matrices for Random Forest, SGD, and SVC on the test set.
5. **ROC Curves** — ROC curves with AUC scores for all three models, plotted against a random-guess baseline.
6. **Random Forest Feature Importance** — horizontal bar chart ranking all 11 features by their importance in the Random Forest model.
7. **Model Comparison Bar Chart** — grouped bar chart comparing Accuracy, AUC, and F1-score (Good class) across all three models.

## Results
- Random Forest delivered the strongest overall performance, with the highest accuracy (94.0%) and AUC (0.9466), making it the most reliable model for distinguishing good wines from average/poor ones in terms of overall discriminative power.
- SGD and SVC traded precision for recall, correctly identifying a higher proportion of actual "good" wines but at the cost of many more false positives, which may be more useful in scenarios where missing a good wine is costlier than incorrectly flagging an average one.
- Alcohol content, sulphates, and volatile acidity emerged consistently as the most influential chemical properties in determining wine quality across both the correlation analysis and the Random Forest feature importance ranking.

## Conclusion
This project demonstrates a complete classification workflow for a class-imbalanced tabular dataset, from identifying and addressing severe class imbalance in the original quality scores, to engineering a more tractable binary target, applying stratified splitting and feature scaling, and comparing three distinct classification algorithms. Based on the actual evaluation results (accuracy and AUC), **Random Forest is the best-performing model** for identifying "good" quality red wines, offering the strongest balance of accuracy, precision, and interpretability through its feature importance rankings. The notebook's own discussion notes that:

- **Random Forest** handles non-linear feature interactions well, is robust to unscaled/skewed features, and provides interpretable feature importances, making it a strong choice for deployment when both interpretability and solid performance matter.
- **SVC** with an RBF kernel often achieves strong AUC on small-to-medium tabular datasets but is less interpretable and slower to tune or scale to larger data.
- **SGD** (a linear model) is the fastest to train and easiest to update online but tends to underperform when quality depends on nonlinear combinations of features (e.g., alcohol interacting with volatile acidity).

The notebook concludes that, for most deployment scenarios, Random Forest offers the best trade-off of performance, robustness, and interpretability, unless inference latency or model size is tightly constrained, in which case a tuned linear model such as SGD may be preferable.

## How to Run This Project
1. Clone or download this repository.
2. Ensure you have Python installed along with the following libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Place the dataset file `winequality-red.csv` in the appropriate directory (update the file path in the notebook if necessary, as the original notebook references a local path).
4. Open the notebook using:
   ```bash
   jupyter notebook Level_no_2_Task_No_2.ipynb
   ```
5. Run all cells sequentially to reproduce the EDA, class imbalance analysis, feature engineering, model training, and evaluation.

## Project Structure
```
├── Level_no_2_Task_No_2.ipynb   # Main Jupyter Notebook containing the full analysis
├── winequality-red.csv          # Dataset used for analysis (not included; path must be set locally)
└── README.md                    # Project documentation
```

## Internship Details
- **Internship Provider:** Oasis Infobyte
- **Domain:** Data Analytics
- **Task Number:** Task 2 (Level 2)

## Author
Prepared as part of a Data Analytics internship task with Oasis Infobyte.
