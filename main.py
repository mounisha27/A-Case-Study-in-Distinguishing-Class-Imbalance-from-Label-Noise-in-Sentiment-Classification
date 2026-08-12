"""
Combined Balancing Comparison — All Methods, Both Models, Before/After Label Fix
==================================================================================
Runs the full grid (None, ROS, RUS, SMOTE, ADASYN, Borderline-SMOTE) x (SVM, NN)
TWICE: once with the original buggy sentiment labeling, once with the fixed
version, then merges both into a single before/after comparison table.
"""

import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# ------------------------------------------------------------
# 1. LOAD + PREPROCESS (shared by both label versions)
# ------------------------------------------------------------

def load_and_preprocess(path="data.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"reviews.text": "reviews", "reviews.username": "username"})

    to_drop = ['dateAdded', 'dateUpdated', 'name', 'asins', 'brand', 'categories',
               'primaryCategories', 'imageURLs', 'keys', 'manufacturer',
               'manufacturerNumber', 'reviews.date', 'reviews.dateAdded',
               'reviews.dateSeen', 'reviews.doRecommend', 'reviews.id',
               'reviews.numHelpful', 'reviews.rating', 'reviews.sourceURLs',
               'reviews.title', 'sourceURLs']
    df.drop(to_drop, inplace=True, axis=1, errors='ignore')

    def remove_punctuations(review):
        for punctuation in string.punctuation:
            review = review.replace(punctuation, '')
        return review

    df['reviews'] = df['reviews'].astype(str).apply(remove_punctuations)
    df['reviews'] = df['reviews'].apply(word_tokenize)

    stemmer = LancasterStemmer()
    df['reviews'] = [[stemmer.stem(w) for w in sentence] for sentence in df['reviews']]

    stop = set(stopwords.words('english'))
    df['reviews'] = df['reviews'].apply(lambda x: [w for w in x if w not in stop])

    return df


# ------------------------------------------------------------
# 2. TWO VERSIONS OF THE LABELING FUNCTION
# ------------------------------------------------------------

def senti_pol_buggy(tokens):
    for word in tokens:
        return TextBlob(word).sentiment.polarity


def senti_pol_fixed(tokens):
    """Fixed version — scores the whole review."""
    full_review = " ".join(tokens)
    return TextBlob(full_review).sentiment.polarity


def assign_labels(df, senti_pol_fn):
    df = df.copy()
    df['senti_polarity'] = df['reviews'].apply(senti_pol_fn)
    condition = [
        df['senti_polarity'] > 0.05,
        (df['senti_polarity'] <= 0.05) & (df['senti_polarity'] > -0.05),
        df['senti_polarity'] <= -0.05
    ]
    values = ['positive', 'neutral', 'negative']
    df['sentiment'] = np.select(condition, values, default='neutral')
    df['reviews_text'] = [" ".join(review) for review in df['reviews']]
    return df


# ------------------------------------------------------------
# 3. RUN THE FULL BALANCING x MODEL GRID FOR ONE LABEL VERSION
# ------------------------------------------------------------

BALANCERS = {
    "None": None,
    "ROS": RandomOverSampler(random_state=0),
    "RUS": RandomUnderSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
}


def run_grid(df, label_version):
    y = df['sentiment']
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['reviews_text'], y, test_size=0.20, random_state=100, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    results = []
    for balance_name, balancer in BALANCERS.items():
        if balancer is None:
            X_tr, y_tr = X_train, y_train
        else:
            X_tr, y_tr = balancer.fit_resample(X_train, y_train)

        for model_name, model in [
            ("SVM", SVC()),
            ("NN", MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300,
                                  activation='relu', solver='adam', random_state=1)),
        ]:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive", "neutral"])
            report = classification_report(y_test, y_pred,
                                            labels=["negative", "positive", "neutral"],
                                            output_dict=True, zero_division=0)

            results.append({
                "Label version": label_version,
                "Model": model_name,
                "Balancing": balance_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Neg Recall": report["negative"]["recall"],
                "Pos Recall": report["positive"]["recall"],
                "Neutral Recall": report["neutral"]["recall"],
                "Macro F1": f1_score(y_test, y_pred, average="macro"),
                "Weighted F1": f1_score(y_test, y_pred, average="weighted"),
            })

    return pd.DataFrame(results)


# ------------------------------------------------------------
# 4. MAIN — RUN BOTH LABEL VERSIONS AND MERGE INTO A COMPARISON TABLE
# ------------------------------------------------------------

def main():
    df_raw = load_and_preprocess("data.csv")

    print("Running grid with BUGGY labels (first-token-only)...")
    df_buggy = assign_labels(df_raw, senti_pol_buggy)
    print("Buggy label distribution:\n", df_buggy['sentiment'].value_counts(), "\n")
    results_buggy = run_grid(df_buggy, "Before (buggy labels)")

    print("Running grid with FIXED labels (whole-review)...")
    df_fixed = assign_labels(df_raw, senti_pol_fixed)
    print("Fixed label distribution:\n", df_fixed['sentiment'].value_counts(), "\n")
    results_fixed = run_grid(df_fixed, "After (fixed labels)")

    combined = pd.concat([results_buggy, results_fixed], ignore_index=True)
    combined.to_csv("balancing_comparison_before_after.csv", index=False)

    # Pivot for a clean side-by-side before/after view per Model x Balancing
    pivot = combined.pivot_table(
        index=["Model", "Balancing"],
        columns="Label version",
        values=["Accuracy", "Neg Recall", "Macro F1"]
    )
    print("\n=== BEFORE / AFTER COMPARISON ===")
    print(pivot.round(3))
    pivot.to_csv("before_after_pivot.csv")

    print("\nSaved: balancing_comparison_before_after.csv, before_after_pivot.csv")


if __name__ == "__main__":
    main()