# Psychological-Condition-Classification
Aim: To develop a system that analyses social media posts to predict psychological conditions such as social anxiety disorder, depression, anxiety, suicidal tendencies, mania, psychosis, substance use disorder, and others using Singular Value Decomposition (SVD)

Dataset: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

Reference Paper - https://www.engr.uvic.ca/~seng474/svd.pdf

### Workflow Overview:
1. Data Acquisition and Loading: Load the dataset from Kaggle into a Pandas DataFrame for evaluation and resolved inconsistencies.

2. Data Preprocessing:
a. Text Cleaning: Remove special characters, numbers, and URLs.
b. Tokenization: Split text into individual words.
c. Stop Words Removal: Remove common words that do not contribute to the sentiment.
d. Stemming/Lemmatization: Reduce words to their base or root form by removing suffixes or other techniques.
Lemmatization provides a greater accuracy compared to stemming, at the cost of computation.

3. Creation of TF-IDF Vector: Convert the cleaned text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization. This allows us to create a sparse matrix, which represents all the posts/textual values in the dataset

4. Dimensionality Reduction using SVD: Since the sparse matrix created using TFIDF Vectorizer would be too huge, it would be computationally expensive to use the entire matrix for any psychological pattern classification task. Hence, we apply SVD to the TF-IDF matrix to reduce dimensionality and capture the most significant features.

5. Model Building and Training: Split the data into training and testing sets. Train multiple machine learning classifiers (e.g., Logistic Regression, Support Vector Machine) using the reduced features via Pipeline.

6. Model Evaluation: Evaluate every model's performance using metrics such as accuracy, precision, recall, and F1-score and finally, obtain the best model suited to the task.
