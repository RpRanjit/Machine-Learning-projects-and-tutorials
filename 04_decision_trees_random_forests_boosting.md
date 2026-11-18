# Machine Learning Course — Phase 5
## Decision Trees, Random Forests & Boosted Trees

---

## 1. Why Trees? (And When to Use Them Over Neural Nets)

Neural networks shine on unstructured data (images, text, audio). For **tabular/structured data** (spreadsheets, database tables — think loan applications, customer churn, sales data), tree-based models usually win: they're often more accurate, need far less data, need no feature scaling, and are much easier to interpret.

**Real-life example we'll use throughout:** predicting whether a loan applicant will default, and separately, the classic Titanic survival dataset.

---

## 2. Decision Trees — The Idea

A decision tree asks a series of yes/no questions about your features, splitting the data at each step, until it reaches a prediction.

```
                Is Age < 18?
               /            \
             Yes             No
              |               |
        Predict: Survived   Is Sex == female?
                            /              \
                          Yes               No
                           |                 |
                    Predict: Survived   Predict: Died
```

**How does it pick which question to ask first?** It picks the split that makes the resulting groups as "pure" as possible (mostly one class), measured by **Gini impurity** or **entropy**.

```
Gini = 1 - Σ(p_i)²        where p_i = proportion of class i in the group
```
Gini = 0 → perfectly pure (all one class). Gini = 0.5 → maximally mixed (50/50 for binary).

```python
import numpy as np

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

# Example: a pure group
print(gini_impurity(np.array([1,1,1,1])))       # 0.0 -- pure
# A mixed group
print(gini_impurity(np.array([1,1,0,0])))       # 0.5 -- max impurity for binary
# A mostly-pure group
print(gini_impurity(np.array([1,1,1,0])))       # 0.375
```

The tree tries every possible split on every feature, picks the one that reduces Gini impurity the most, and repeats recursively for each resulting branch.

---

## 3. Decision Trees with scikit-learn — Real Dataset (Titanic)

```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Titanic dataset — built into seaborn, real historical passenger data
df = sns.load_dataset('titanic')
print(df.head())

# Keep relevant columns, handle missing values
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})   # trees need numbers, not strings

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# No scaling needed! Trees split on raw thresholds, unaffected by feature scale.
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# Visualize the actual tree — one of the few ML models you can literally read
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X.columns, class_names=['Died','Survived'],
          filled=True, rounded=True, fontsize=10)
plt.savefig('titanic_tree.png', dpi=150)
plt.show()

# Feature importance
importance = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)
```

**`max_depth` controls overfitting directly** — an unlimited-depth tree will keep splitting until every leaf is pure, essentially memorizing the training set (classic overfitting). Try `max_depth=None` vs `max_depth=3` and compare train vs test accuracy.

---

## 4. Random Forests — Many Trees Are Better Than One

**The problem with a single tree:** high variance — small changes in the data can produce a completely different tree.

**The fix (bagging = bootstrap aggregating):** train many trees, each on a random subset of the data AND a random subset of features, then average their votes.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=200,      # number of trees
    max_depth=6,
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Feature importance is usually more reliable than a single tree's
importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)
```

**Why it works — intuition:** imagine asking 200 slightly-different-minded people to each guess a number, then averaging their guesses. Individual errors tend to cancel out; the wisdom of the crowd beats any single overconfident tree.

**Compare single tree vs forest directly:**
```python
print(f"Single tree test accuracy: {accuracy_score(y_test, tree.predict(X_test)):.3f}")
print(f"Random forest test accuracy: {accuracy_score(y_test, forest.predict(X_test)):.3f}")
```
The forest is almost always more accurate and more stable.

---

## 5. Boosted Trees (Gradient Boosting / XGBoost)

**Random forests** build trees independently and average them (parallel, reduces variance).
**Boosting** builds trees sequentially — each new tree specifically tries to fix the mistakes of the previous ones (reduces bias, often the most accurate approach for tabular data).

```bash
pip install xgboost --break-system-packages
```

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,      # how much each new tree corrects previous errors
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Compare all three models side by side
print("\n--- Model Comparison ---")
print(f"Decision Tree:  {accuracy_score(y_test, tree.predict(X_test)):.3f}")
print(f"Random Forest:  {accuracy_score(y_test, forest.predict(X_test)):.3f}")
print(f"XGBoost:        {accuracy_score(y_test, xgb_model.predict(X_test)):.3f}")
```

**How boosting works, step by step:**
1. Train a small tree on the data. It makes some errors.
2. Train a second tree specifically on the *residual errors* of the first.
3. Add its predictions (scaled by `learning_rate`) to the first tree's.
4. Repeat for `n_estimators` rounds — each tree nudges the total prediction closer to correct.

This is why XGBoost wins the majority of Kaggle tabular-data competitions — it's very good at squeezing out the last bit of accuracy.

---

## 6. PROJECT 4: Model Comparison on a Real Regression Task (California Housing)

Go back to the California Housing dataset from Phase 2, but this time compare Linear Regression against tree-based models:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
X = data.frame.drop('MedHouseVal', axis=1)
y = data.frame['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"{name}: MSE={mse:.3f}, R²={r2:.3f}")

# Visualize the comparison
import matplotlib.pyplot as plt
names = list(results.keys())
r2_scores = [results[n]['R2'] for n in names]

plt.figure(figsize=(8,5))
plt.bar(names, r2_scores, color=['steelblue','forestgreen','darkorange'])
plt.ylabel('R² Score (higher = better)')
plt.title('Model Comparison: California Housing Price Prediction')
plt.ylim(0, 1)
plt.savefig('model_comparison.png')
plt.show()
```

**Expected result:** you'll see XGBoost and Random Forest noticeably beat plain Linear Regression on R² — trees can capture non-linear relationships (e.g. "price rises with income, but not in a straight line") that a linear model structurally can't.

**Extend it yourself:**
1. Tune `max_depth` and `n_estimators` for the Random Forest — plot R² vs `max_depth` from 2 to 20. Where does it start overfitting (test R² drops while train R² keeps climbing)?
2. Try `GridSearchCV` to automatically search over `n_estimators`, `max_depth`, and `learning_rate` for XGBoost.
3. Plot XGBoost's feature importances — do they match what a real estate agent would tell you matters?

---

## 7. Quick Decision Guide: Which Model When?

| Situation | Try first |
|---|---|
| Tabular/structured data (spreadsheets, CSVs) | Random Forest or XGBoost |
| Images, audio, video, raw text | Neural networks (CNNs, transformers) |
| Need to explain every decision (legal, medical) | Single Decision Tree or Logistic Regression |
| Small dataset (few hundred rows) | Linear/Logistic Regression, or shallow trees |
| Need the absolute best accuracy on tabular data, don't care about interpretability | XGBoost / LightGBM |
| Very large dataset, complex patterns | Neural networks |

---

## What's next

**Phase 6** — the final phase — covers **ML Development Practices**: how real ML systems are built and evaluated in industry (train/val/test splits done right, error analysis, handling imbalanced data, cross-validation, hyperparameter tuning, avoiding data leakage, and deploying a model), plus a capstone project that ties everything together.

Say **"continue to phase 6"** when ready.
