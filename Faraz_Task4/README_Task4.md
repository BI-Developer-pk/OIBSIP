# Vaccine Sentiment Analysis on Twitter Data (NLP + Machine Learning Classification)

## Overview
This project applies Natural Language Processing (NLP) and machine learning techniques to classify the sentiment of tweets related to vaccines. The notebook covers text preprocessing, TF-IDF feature extraction, training and evaluating two classification models (Naive Bayes and Logistic Regression), and performing error analysis on misclassified tweets.

## Objective
The objective of this project is to build and compare machine learning models that classify vaccine-related tweets into Positive, Neutral, or Negative sentiment categories, and to analyze model performance and misclassifications to understand the strengths and limitations of each approach.

## Dataset Description
- **Dataset Name:** `label.csv`
- **Key Columns:**
  - `tweet_id` — unique identifier for each tweet
  - `safe_text` — the tweet text content
  - `label` — numeric sentiment label (`1.0` = Positive, `0.0` = Neutral, `-1.0` = Negative)
  - `agreement` — a column present in the raw data, though it was not used in the analysis
- **Class Distribution (after mapping labels to sentiment categories and dropping missing values):**
  - Neutral: 4,908 tweets
  - Positive: 4,053 tweets
  - Negative: 1,038 tweets

The dataset does not specify its original source; it was loaded directly from a local CSV file for analysis. The content of the tweets (e.g., references to MMR vaccines, autism, and immunizations) indicates the dataset centers on vaccine-related public discourse.

## Tools and Technologies Used
- **Language:** Python
- **Libraries:**
  - `pandas` — data manipulation and analysis
  - `numpy` — numerical computations
  - `matplotlib` — data visualization
  - `seaborn` — statistical data visualization
  - `nltk` — text preprocessing (stopwords, tokenization, lemmatization)
  - `scikit-learn` — TF-IDF feature extraction, train/test splitting, model training, and evaluation (`TfidfVectorizer`, `train_test_split`, `MultinomialNB`, `LogisticRegression`, `classification_report`, `confusion_matrix`, `accuracy_score`)
- **Environment:** Jupyter Notebook

## Project Workflow

### Data Collection
The dataset was loaded from a local CSV file (`label.csv`) using `pandas.read_csv()`.

### Data Cleaning
- Selected only the relevant columns (`safe_text` and `label`) and dropped rows with missing values in either column.
- Mapped the numeric `label` column to a human-readable `sentiment` column (`Positive`, `Neutral`, `Negative`).
- After text preprocessing, re-checked for and dropped any remaining missing values in `cleaned_text` and `sentiment` before modeling.

### Data Preprocessing
A custom text preprocessing function (`preprocess_tweet`) was applied to the raw tweet text (`safe_text`) to produce a cleaned version (`cleaned_text`). The preprocessing steps included:
- Replacing HTML entity `&amp;` with "and" and removing the `<user>` placeholder tag.
- Converting text to lowercase.
- Removing punctuation.
- Tokenizing the text using NLTK's `word_tokenize`.
- Removing English stopwords using NLTK's stopword list.
- Lemmatizing remaining tokens using NLTK's `WordNetLemmatizer`.

Following preprocessing, the data was split into training and testing sets (80/20 split, `random_state=42`, stratified by sentiment class) and transformed into numerical features using `TfidfVectorizer` with a maximum of 5,000 features.

### Exploratory Data Analysis
- Reviewed the most frequent tweet texts using `value_counts()` to understand recurring content (e.g., repeated tweets about vaccine-autism studies).
- Examined the overall class distribution of sentiment labels (Neutral, Positive, Negative) both numerically and visually.

### Visualization
Charts were created using `matplotlib` and `seaborn`, including a count plot of sentiment distribution and confusion matrix heatmaps for each trained model (detailed in the **Visualizations Performed** section below).

### Modeling
Two supervised classification models were trained on the TF-IDF-transformed text data to predict tweet sentiment:

1. **Multinomial Naive Bayes** (`MultinomialNB`)
2. **Logistic Regression** (`LogisticRegression`, `max_iter=1000`)

Both models were trained on the same training set (`X_train_tfidf`, `y_train`) and evaluated on the same held-out test set (`X_test_tfidf`, `y_test`).

### Evaluation
Each model was evaluated using accuracy, a full classification report (precision, recall, f1-score per class), and a confusion matrix:

**Naive Bayes:**
- Accuracy: **70.15%**
- Negative: precision 0.70, recall 0.03, f1-score 0.06
- Neutral: precision 0.77, recall 0.77, f1-score 0.77
- Positive: precision 0.64, recall 0.78, f1-score 0.70
- Macro average f1-score: 0.51

**Logistic Regression:**
- Accuracy: **73.25%**
- Negative: precision 0.67, recall 0.22, f1-score 0.33
- Neutral: precision 0.78, recall 0.81, f1-score 0.79
- Positive: precision 0.69, recall 0.77, f1-score 0.73
- Macro average f1-score: 0.62

An error analysis was also performed on the Logistic Regression model's predictions, examining specific examples of misclassified tweets to understand where the model struggled (e.g., tweets with sarcasm, mixed emotional tone, or ambiguous context).

## Key Insights and Findings
- The dataset is imbalanced, with Neutral (4,908) and Positive (4,053) tweets substantially outnumbering Negative tweets (1,038).
- Both models achieved reasonably strong overall accuracy (70.15% for Naive Bayes, 73.25% for Logistic Regression), but both struggled significantly with the minority **Negative** class.
- Naive Bayes had extremely poor recall on the Negative class (0.03), meaning it correctly identified only a tiny fraction of actual negative tweets, likely due to class imbalance and its feature-independence assumption.
- Logistic Regression showed notably better (though still limited) performance on the Negative class (recall of 0.22), and achieved a higher macro-average f1-score (0.62 vs. 0.51), indicating more balanced performance across all three sentiment classes.
- Error analysis revealed real examples of misclassification, such as a Positive tweet about a vaccination event being predicted as Neutral, and a Negative tweet expressing frustration about immunization costs being predicted as Positive — illustrating the difficulty of capturing sarcasm, frustration, and nuanced context using this modeling approach.
- The notebook's conclusion section contains a placeholder sentence intended for the user to fill in with final accuracy and macro F1-score values; this was not completed with specific values in the notebook itself, though the actual metrics are available in the model evaluation output described above.

## Visualizations Performed
1. **Vaccine Sentiment Distribution** — count plot showing the number of tweets in each sentiment category (Positive, Neutral, Negative).
2. **Confusion Matrix — Naive Bayes** — heatmap of actual vs. predicted labels for the Naive Bayes model.
3. **Confusion Matrix — Logistic Regression** — heatmap of actual vs. predicted labels for the Logistic Regression model.

## Results
- Logistic Regression outperformed Naive Bayes on this task, achieving higher overall accuracy (73.25% vs. 70.15%) and a notably higher macro-average f1-score (0.62 vs. 0.51).
- Both models performed well on the Neutral and Positive classes but were substantially weaker at correctly identifying Negative tweets, with Naive Bayes performing especially poorly on this minority class.
- Error analysis confirmed that misclassifications often involved tweets with subtle emotional tone, sarcasm, or context that is difficult to capture through TF-IDF-based bag-of-words representations alone.

## Conclusion
The notebook concludes that **Logistic Regression is the better-performing model** for this vaccine sentiment classification task. This is attributed to three main factors described in the notebook:

1. **Handling of Sparse TF-IDF Vectors** — Logistic Regression is effective at finding optimal decision boundaries by assigning precise weights to the high-dimensional, sparse features produced by TF-IDF.
2. **Feature Interdependence** — Naive Bayes assumes complete independence between words, which does not hold well for natural language, whereas Logistic Regression better captures relationships between correlated terms.
3. **Mitigating Class Imbalance** — Naive Bayes tends to over-classify toward the majority class in imbalanced datasets, while Logistic Regression's built-in L2 regularization helps produce more balanced performance across classes.

The notebook also discusses potential **real-world applications** of a vaccine-focused sentiment analysis pipeline, including public health monitoring — for example, government agencies or organizations such as the WHO using the model to track social media sentiment and detect emerging vaccine hesitancy trends by geography. The notebook's discussion of real-world applications extends beyond this point, but the additional content was not fully captured in the analyzed cells.

## How to Run This Project
1. Clone or download this repository.
2. Ensure you have Python installed along with the following libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter
   ```
3. Download the required NLTK resources (these are referenced as commented-out download commands in the notebook and should be run once if not already installed):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('wordnet')
   ```
4. Place the dataset file `label.csv` in the appropriate directory (update the file path in the notebook if necessary, as the original notebook references a local path).
5. Open the notebook using:
   ```bash
   jupyter notebook Level_no_1_Task_No_4.ipynb
   ```
6. Run all cells sequentially to reproduce the preprocessing, model training, evaluation, and error analysis.

## Project Structure
```
├── Level_no_1_Task_No_4.ipynb     # Main Jupyter Notebook containing the full analysis
├── label.csv                      # Dataset used for analysis (not included; path must be set locally)
└── README.md                      # Project documentation
```

## Internship Details
- **Internship Provider:** Oasis Infobyte
- **Domain:** Data Analytics
- **Task Number:** Task 4 (Level 1)

## Author
Prepared as part of a Data Analytics internship task with Oasis Infobyte.
