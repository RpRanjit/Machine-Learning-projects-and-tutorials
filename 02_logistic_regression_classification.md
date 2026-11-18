# Machine Learning Course — Phase 3
## Logistic Regression, Classification, Overfitting & Regularization

---

## 1. From Regression to Classification

Linear regression predicts a number (price, temperature). **Classification** predicts a category: spam/not spam, cancer/benign, cat/dog.

**Real-life example we'll use:** predicting whether a tumor is malignant or benign from measurements (the classic Wisconsin Breast Cancer dataset — real medical data, built into sklearn).

---

## 2. The Sigmoid Function

Linear regression output can be any number (-∞ to +∞). We need output between 0 and 1 (a probability). The **sigmoid function** squashes any number into that range:

```
g(z) = 1 / (1 + e^(-z))
```

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.axhline(0.5, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Function: squashes any number to (0,1)')
plt.savefig('sigmoid.png')
plt.show()

print(sigmoid(0))    # 0.5  -> decision boundary
print(sigmoid(5))    # 0.99 -> confident "yes"
print(sigmoid(-5))   # 0.01 -> confident "no"
```

**Logistic Regression model:**
```
f(x) = sigmoid(w·x + b)
```
If `f(x) >= 0.5` → predict class 1. If `f(x) < 0.5` → predict class 0.

---

## 3. The Cost Function for Classification (Log Loss / Binary Cross-Entropy)

MSE doesn't work well for classification (it creates a wavy, non-convex cost surface). Instead we use **log loss**:

```
J(w,b) = -(1/m) * Σ [ y*log(f(x)) + (1-y)*log(1-f(x)) ]
```

Intuition: if the true label is 1 and the model predicts 0.99, cost is tiny. If it predicts 0.01, cost is huge. It punishes confident wrong answers hard.

---

## 4. Logistic Regression — From Scratch

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip avoids overflow

def compute_cost(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    f = sigmoid(z)
    epsilon = 1e-15  # avoid log(0)
    cost = -np.mean(y*np.log(f+epsilon) + (1-y)*np.log(1-f+epsilon))
    return cost

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    f = sigmoid(z)
    error = f - y
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b))
        if i % 200 == 0:
            print(f"Iter {i}: cost {cost_history[-1]:.4f}")
    return w, b, cost_history

# Toy example: hours studied vs pass/fail exam
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=float)
y = np.array([0,0,0,0,1,0,1,1,1,1], dtype=float)

# Normalize feature (important for gradient descent to converge well)
X_norm = (X - X.mean()) / X.std()

w_init = np.zeros(1)
b_init = 0.0
w, b, costs = gradient_descent(X_norm, y, w_init, b_init, alpha=0.5, num_iters=1000)

# Predict: will someone who studies 6 hours pass?
test_hours = np.array([[6]])
test_norm = (test_hours - X.mean()) / X.std()
prob = sigmoid(np.dot(test_norm, w) + b)
print(f"\nProbability of passing with 6 hrs study: {prob[0]:.2f}")
print(f"Prediction: {'PASS' if prob[0]>=0.5 else 'FAIL'}")
```

---

## 5. Logistic Regression with scikit-learn — Real Dataset

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 0 = malignant, 1 = benign

print(X.shape)         # (569, 30) -- 30 real medical measurements
print(data.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nFull report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

**Reading the confusion matrix (critical real-world skill):**
```
                Predicted Malignant   Predicted Benign
Actual Malignant      TP                    FN   <- False Negative is DANGEROUS (missed cancer)
Actual Benign         FP                    TN
```
In medical diagnosis, a **false negative** (telling a sick patient they're fine) is usually far worse than a false positive. This is why accuracy alone isn't enough — you look at **recall** for the malignant class specifically.

---

## 6. Overfitting vs Underfitting

- **Underfitting (high bias):** model too simple, does poorly on both training AND test data. (e.g. fitting a straight line to clearly curved data)
- **Overfitting (high variance):** model memorizes training data, does great on training but poorly on new data. (e.g. a very high-degree polynomial that wiggles through every single point)
- **Just right:** generalizes well to new, unseen data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate noisy curved data
np.random.seed(0)
X = np.linspace(0, 1, 20).reshape(-1,1)
y = np.sin(2*np.pi*X).ravel() + np.random.normal(0, 0.2, 20)

X_test = np.linspace(0, 1, 100).reshape(-1,1)

degrees = [1, 4, 15]   # underfit, good fit, overfit
titles = ['Underfitting (degree 1)', 'Good fit (degree 4)', 'Overfitting (degree 15)']

plt.figure(figsize=(15,4))
for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X_test)

    plt.subplot(1,3,i+1)
    plt.scatter(X, y, color='red', label='Training data')
    plt.plot(X_test, y_pred, color='blue', label='Model')
    plt.plot(X_test, np.sin(2*np.pi*X_test), color='green', linestyle='--', label='True function')
    plt.ylim(-2,2)
    plt.title(titles[i])
    plt.legend()

plt.tight_layout()
plt.savefig('overfitting_demo.png')
plt.show()
```

Run this — you'll SEE the degree-15 curve wildly wiggling to hit every training point while missing the true pattern. This picture is worth more than any explanation.

---

## 7. Regularization — Fixing Overfitting

Regularization adds a penalty for large weights, forcing the model to stay simpler.

```
J(w,b) = [normal cost] + (λ/2m) * Σ(w_j²)      <- L2 (Ridge)
J(w,b) = [normal cost] + (λ/2m) * Σ|w_j|        <- L1 (Lasso)
```
`λ` (lambda) controls the strength: `λ=0` → no regularization (may overfit). `λ` too large → underfitting (model becomes too simple / weights pushed near zero).

```python
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 1, 20).reshape(-1,1)
y = np.sin(2*np.pi*X).ravel() + np.random.normal(0, 0.2, 20)
X_test = np.linspace(0, 1, 100).reshape(-1,1)

models = {
    'No regularization (overfit)': make_pipeline(PolynomialFeatures(15), LinearRegression()),
    'Ridge (L2), alpha=0.001': make_pipeline(PolynomialFeatures(15), Ridge(alpha=0.001)),
    'Ridge (L2), alpha=1.0': make_pipeline(PolynomialFeatures(15), Ridge(alpha=1.0)),
}

plt.figure(figsize=(15,4))
for i, (name, model) in enumerate(models.items()):
    model.fit(X, y)
    y_pred = model.predict(X_test)
    plt.subplot(1,3,i+1)
    plt.scatter(X, y, color='red')
    plt.plot(X_test, y_pred, color='blue')
    plt.ylim(-2,2)
    plt.title(name)

plt.tight_layout()
plt.savefig('regularization_demo.png')
plt.show()
```

Notice: as `alpha` (λ) increases, the wild degree-15 curve smooths out and starts looking like the true sine wave again — same model, just penalized for being unnecessarily wiggly.

**In classification**, use `LogisticRegression(C=...)` — note sklearn uses `C = 1/λ`, so **smaller C = stronger regularization** (opposite direction, easy to mix up).

---

## 8. Diagnosing Bias vs Variance in Practice

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training error')
    plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation error')
    plt.xlabel('Training set size')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
```

**Reading it:**
- Both errors high, close together → **underfitting** (high bias). Fix: bigger/more complex model, more features, less regularization.
- Big gap between low training error and high validation error → **overfitting** (high variance). Fix: more data, regularization, simpler model, fewer features.

---

## 9. PROJECT 2: Breast Cancer Classifier with Regularization Tuning

Build this end to end:

1. Load `load_breast_cancer()`, split train/test, scale features.
2. Train `LogisticRegression` with default settings — get baseline accuracy.
3. Try `C = [0.001, 0.01, 0.1, 1, 10, 100]` in a loop, plot accuracy vs C (log x-axis).
4. Pick the best `C` based on validation performance (use `cross_val_score`, not the test set, to choose it).
5. Report final test accuracy, confusion matrix, and precision/recall for the malignant class specifically.
6. **Bonus:** which 5 features have the largest absolute coefficients? Do they make medical sense (e.g. "mean radius", "mean concave points")?

```python
from sklearn.model_selection import cross_val_score
import numpy as np

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
for C in C_values:
    model = LogisticRegression(C=C, max_iter=5000)
    scores = cross_val_score(model, X_train_s, y_train, cv=5)
    print(f"C={C}: mean CV accuracy = {scores.mean():.4f}")
```

---

## What's next

**Phase 4** covers **Neural Networks & TensorFlow** — building your first deep learning model, multiclass classification with softmax, and a real image-classification project (handwritten digits).

Say **"continue to phase 4"** when ready.
