"""
Iris Flower Classification
--------------------------
Classify iris flowers into Setosa, Versicolor and Virginica
from sepal and petal measurements.

Steps:
  1. Load the dataset
  2. Explore it visually (scatter plots, histograms)
  3. Split into training / test sets
  4. Preprocess (feature scaling)
  5. Train classifiers (Logistic Regression, KNN, Decision Tree)
  6. Evaluate (accuracy, precision, recall, F1, confusion matrix)

Run:  python iris_classification.py
All plots are saved in the "outputs/" folder.
"""

import os

import matplotlib
matplotlib.use("Agg")  # save plots to files (works without a display)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- 1. LOAD
print("=" * 60)
print("STEP 1: Load the dataset")
print("=" * 60)
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print(f"Shape: {df.shape}  (150 flowers, 4 features + target)")
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSamples per species:")
print(df["species"].value_counts())
print("\nSummary statistics:")
print(df.describe().round(2))


# ------------------------------------------------------------- 2. EXPLORE
print("\n" + "=" * 60)
print("STEP 2: Explore visually")
print("=" * 60)
sns.set_theme(style="whitegrid")
features = iris.feature_names

# 2a. Histograms of every feature, coloured by species
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feature in zip(axes.ravel(), features):
    sns.histplot(data=df, x=feature, hue="species", kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_histograms.png", dpi=120)
plt.close()

# 2b. Scatter plot: petal length vs petal width
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, x="petal length (cm)", y="petal width (cm)",
    hue="species", style="species", s=70,
)
plt.title("Petal length vs Petal width")
plt.savefig(f"{OUTPUT_DIR}/02_scatter_petal.png", dpi=120)
plt.close()

# 2c. Pair plot of all features
pair = sns.pairplot(df.drop(columns="target"), hue="species", corner=True)
pair.savefig(f"{OUTPUT_DIR}/03_pairplot.png", dpi=100)
plt.close("all")

# 2d. Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation.png", dpi=120)
plt.close()
print("Saved: histograms, scatter plot, pair plot, correlation heatmap")


# ---------------------------------------------------------------- 3. SPLIT
print("\n" + "=" * 60)
print("STEP 3: Split into training and test sets (80 / 20)")
print("=" * 60)
X = df[features]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")


# ------------------------------------------------------------ 4. PREPROCESS
print("\n" + "=" * 60)
print("STEP 4: Preprocess (standardize features)")
print("=" * 60)
print("The data has no missing values, so only scaling is needed.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on TRAIN only
X_test_scaled = scaler.transform(X_test)        # reuse train statistics
print("Scaler fitted on training data only (avoids data leakage).")


# ----------------------------------------------------------------- 5. TRAIN
print("\n" + "=" * 60)
print("STEP 5: Train classifiers")
print("=" * 60)
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"Trained: {name}")


# -------------------------------------------------------------- 6. EVALUATE
print("\n" + "=" * 60)
print("STEP 6: Evaluate on the test set")
print("=" * 60)
results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_scaled)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1-score": f1_score(y_test, y_pred, average="macro"),
    })
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=iris.target_names, cmap="Blues", ax=ax
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_confusion_matrices.png", dpi=120)
plt.close()

results_df = pd.DataFrame(results).set_index("Model").round(3)
print("\nModel comparison (macro-averaged):")
print(results_df)
results_df.to_csv(f"{OUTPUT_DIR}/results.csv")

best = results_df["Accuracy"].idxmax()
print(f"\nBest model on this test split: {best} "
      f"(accuracy = {results_df.loc[best, 'Accuracy']:.3f})")
print("\nNote: Iris is small (150 samples) and well separated, so high scores")
print("are expected. Results will not transfer to harder real-world data.")

# Predict a new flower
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=features)
pred = models[best].predict(scaler.transform(sample))[0]
print(f"\nExample: flower {sample.values[0].tolist()} -> predicted "
      f"'{iris.target_names[pred]}' using {best}")
print(f"\nAll plots saved in '{OUTPUT_DIR}/'. Done!")
