# Machine Learning Course — Phase 6 (Final)
## ML Development Practices & Capstone Project

---

## 1. The Real ML Workflow (What Actually Happens on the Job)

```
1. Define the problem & success metric
2. Collect & explore data (EDA)
3. Clean & preprocess data
4. Split data (train/val/test) — do this EARLY, before any tuning
5. Train baseline model
6. Error analysis — what is it getting wrong, and why?
7. Iterate: feature engineering, try new models, tune hyperparameters
8. Final evaluation on the untouched test set
9. Deploy & monitor
```

The biggest beginner mistake: jumping straight to step 5-7 and skipping steps 1, 4, and 6. Let's fix that.

---

## 2. Train / Validation / Test — Doing It Right

```python
from sklearn.model_selection import train_test_split

# First split: separate out the test set — LOCK IT AWAY, don't touch until the very end
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Second split: train vs validation (used for tuning)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
# 0.176 * 0.85 ≈ 0.15, giving roughly 70/15/15 train/val/test

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

**Why three sets, not two?**
- **Train:** the model learns from this.
- **Validation:** you tune hyperparameters and pick between models using this. You look at it many times.
- **Test:** touched exactly ONCE, at the very end, to report the honest final number. If you tune based on test performance, you've secretly leaked information and your reported accuracy is a lie (to yourself).

---

## 3. Cross-Validation — Getting a More Reliable Estimate

A single train/val split can be lucky or unlucky. **K-Fold Cross-Validation** splits data into K parts, trains K times (each part takes a turn as validation), and averages the results.

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

print(f"Fold scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

The standard deviation across folds tells you how stable your model is — a small std means consistent performance regardless of which slice of data it sees.

---

## 4. Hyperparameter Tuning — Systematically, Not by Guessing

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
```

`GridSearchCV` tries every combination (3×4×3 = 36 combos × 5 folds = 180 model fits here). For bigger search spaces, use `RandomizedSearchCV` — it samples randomly instead of exhaustively, much faster with almost as good results.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=30, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

---

## 5. Handling Imbalanced Data

**Real-world problem:** fraud detection where 99.9% of transactions are legit. A model that always predicts "not fraud" gets 99.9% accuracy while being completely useless.

```python
import pandas as pd
from sklearn.datasets import make_classification

# Simulate imbalanced data: 950 legit, 50 fraud
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=42)
print(pd.Series(y).value_counts())
```

**Fixes:**

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 1. Use the RIGHT metrics — never accuracy alone for imbalanced data
y_pred = model.predict(X_test)
print(f"Precision: {precision_score(y_test, y_pred):.3f}")  # of predicted fraud, how much was real?
print(f"Recall: {recall_score(y_test, y_pred):.3f}")        # of real fraud, how much did we catch?
print(f"F1: {f1_score(y_test, y_pred):.3f}")                # harmonic mean of precision & recall
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")

# 2. Class weighting — penalize mistakes on the minority class more
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# 3. Resampling — SMOTE creates synthetic minority-class examples
# pip install imbalanced-learn --break-system-packages
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(pd.Series(y_resampled).value_counts())  # now balanced
```

**Precision vs Recall trade-off — pick based on the real cost of errors:**
- Fraud detection: prioritize **recall** (missing real fraud is costly) even if it means more false alarms.
- Spam filter: prioritize **precision** (marking a real email as spam is annoying — you'd rather let a few spam through).

---

## 6. Avoiding Data Leakage (The Silent Killer of ML Projects)

Data leakage = information from outside the training set sneaks in, making your model look great in testing but fail in the real world.

**Common leakage mistakes:**
```python
# WRONG: fitting the scaler on ALL data before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)              # sees test data statistics!
X_train, X_test = train_test_split(X_scaled, ...)

# RIGHT: fit scaler ONLY on training data, then apply to test
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled = scaler.transform(X_test)          # transform test using train's stats
```

**Other classic leakage examples:**
- Including a feature that wouldn't be available at prediction time (e.g., "days until cancellation" when predicting churn).
- Doing feature selection or imputation on the whole dataset before splitting.
- Time-series data: randomly shuffling rows instead of splitting chronologically (you'd be "predicting the past using the future").

**Best practice: use `Pipeline` so preprocessing is always correctly scoped to training folds:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Now cross_val_score automatically fits the scaler correctly on each fold
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

---

## 7. CAPSTONE PROJECT: End-to-End Customer Churn Prediction

Bring together everything from all 6 phases into one real project. Use the **Telco Customer Churn dataset** (a genuinely used business dataset — predicting which customers will cancel their subscription).

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load data
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
df = pd.read_csv('telco_churn.csv')

# STEP 2: Explore
print(df.info())
print(df['Churn'].value_counts(normalize=True))  # check class imbalance
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df.isnull().sum())

# STEP 3: Clean
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# STEP 4: Split BEFORE any preprocessing
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# stratify=y keeps the same churn ratio in both sets -- important for imbalanced data

# STEP 5: Build a preprocessing + model pipeline (avoids leakage automatically)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, scale_pos_weight=3, random_state=42, eval_metric='logloss')
}

# STEP 6: Train, cross-validate, and compare
results = {}
for name, clf in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]
    test_auc = roc_auc_score(y_test, y_proba)

    results[name] = {'cv_auc_mean': cv_scores.mean(), 'test_auc': test_auc, 'pipeline': pipe}
    print(f"\n{name}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Test ROC-AUC: {test_auc:.3f}")
    print(classification_report(y_test, y_pred))

# STEP 7: Pick the best model, do final error analysis
best_name = max(results, key=lambda k: results[k]['test_auc'])
best_pipe = results[best_name]['pipeline']
print(f"\nBest model: {best_name}")

y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay','Churn'], yticklabels=['Stay','Churn'])
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title(f'Confusion Matrix — {best_name}')
plt.savefig('churn_confusion_matrix.png')
plt.show()

# STEP 8: Business interpretation
# Which factors drive churn the most? Use the tree-based model's feature importance,
# or coefficients if Logistic Regression is best
print("\nBusiness insight: identify top churn drivers and suggest 2-3 retention actions")
```

**Full capstone checklist — do all of these:**
1. ✅ EDA: plot churn rate by contract type, tenure, monthly charges (`sns.boxplot`, `sns.countplot`)
2. ✅ Clean data properly, split before preprocessing
3. ✅ Build a `Pipeline` to prevent leakage
4. ✅ Compare 3 models with cross-validation
5. ✅ Handle the class imbalance (churn is usually the minority class)
6. ✅ Tune the best model with `GridSearchCV`
7. ✅ Report the RIGHT metrics (ROC-AUC, recall for churners — missing a churner costs more than a false alarm)
8. ✅ Do error analysis: look at 10 misclassified customers — what do they have in common?
9. ✅ Write a 3-sentence business summary: what drives churn, and what should the company do about it?

---

## 8. Deployment — The Basic Idea (So You Know What's Next)

Once you have a model you trust:

```python
import joblib

# Save the trained pipeline (preprocessing + model together)
joblib.dump(best_pipe, 'churn_model.pkl')

# Later, in production:
loaded_model = joblib.load('churn_model.pkl')
new_customer = pd.DataFrame([{...}])  # a single new customer's data
prediction = loaded_model.predict(new_customer)
probability = loaded_model.predict_proba(new_customer)[:,1]
```

In real companies, this gets wrapped in an API (commonly with FastAPI or Flask), monitored for **data drift** (does new data still look like training data?) and **model decay** (does accuracy degrade over time as the world changes?), and retrained periodically.

---

## Course Complete — What You've Built

| Phase | Concepts | Project |
|---|---|---|
| 1-2 | Linear regression, gradient descent, cost functions | House price predictor |
| 3 | Logistic regression, overfitting, regularization | Cancer classifier |
| 4 | Neural networks, TensorFlow, softmax | MNIST digit recognizer |
| 5 | Decision trees, random forests, XGBoost | Model comparison suite |
| 6 | CV, tuning, imbalanced data, leakage, deployment | End-to-end churn predictor |

**Where to go from here (pick based on interest):**
- **Deep learning specialization:** CNNs (images), RNNs/Transformers (text/sequences) — deeplearning.ai
- **MLOps:** deploying and monitoring models in production
- **Kaggle competitions:** apply everything here to real competitions, learn from top solutions
- **Specialize:** NLP, computer vision, recommender systems, or time-series forecasting

You now have the full toolkit — from a straight line through gradient descent, to neural networks, to the tree ensembles that win most real-world tabular competitions, to how to actually validate and ship a model responsibly. Build things with it.
