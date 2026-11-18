# Machine Learning Course — Phase 1 & 2
## Foundations + Linear Regression + Multiple Linear Regression

---

## 1. Setup

Install everything you need once:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn --break-system-packages
```

Create a folder `ml_course/` and do all your work there. Use Jupyter Notebook or VS Code — both fine.

---

## 2. NumPy — The Language of ML

Every ML model is really just matrix math. NumPy is how Python does matrix math fast.

```python
import numpy as np

# Vectors and matrices
x = np.array([1, 2, 3])          # a vector (1D array)
X = np.array([[1, 2], [3, 4]])   # a matrix (2D array)

print(x.shape)   # (3,)
print(X.shape)   # (2, 2)

# Dot product — the single most important operation in ML
w = np.array([0.5, -0.2, 0.1])
b = 2.0
prediction = np.dot(x, w) + b
print(prediction)   # 2.4

# Vectorized operations (NO loops — this is 100x faster than Python loops)
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
print(a + b)        # [11 22 33 44]
print(a * b)        # [10 40 90 160]
print(np.mean(a))   # 2.5
print(np.sum(a))    # 10
```

**Real-life analogy:** if `x` is a house's features (size, bedrooms, age) and `w` is how much each feature matters, `np.dot(x, w) + b` IS the linear regression prediction. That's it — that's the whole model.

**Practice:** create a vector of 5 numbers, compute its mean and standard deviation using `np.mean()` and `np.std()`.

---

## 3. Linear Regression — From Scratch (no libraries)

### The idea
You have data points. You want to draw the "best" straight line through them so you can predict new values.

**Real-life example:** predicting a house's price from its size (in square feet).

| Size (sqft) | Price ($1000s) |
|---|---|
| 1000 | 200 |
| 1500 | 300 |
| 2000 | 400 |
| 2500 | 480 |

The model: `price = w * size + b`

### The Cost Function (Mean Squared Error)
Measures how wrong the line is. Lower = better.

```
J(w,b) = (1/2m) * Σ (f(x_i) - y_i)²
```
where `f(x_i) = w*x_i + b`, `m` = number of examples.

### Gradient Descent
An algorithm that nudges `w` and `b` in the direction that reduces the cost, step by step.

```
w = w - α * (∂J/∂w)
b = b - α * (∂J/∂b)
```
`α` (alpha) = learning rate — how big each step is.

### Full code — implemented from scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# Data: house size (sqft, scaled to 1000s) vs price ($1000s)
X = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
y = np.array([200, 300, 400, 480, 560, 620])

def compute_cost(X, y, w, b):
    m = len(X)
    predictions = w * X + b
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    m = len(X)
    predictions = w * X + b
    dj_dw = np.sum((predictions - y) * X) / m
    dj_db = np.sum(predictions - y) / m
    return dj_dw, dj_db

def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    w, b = w_init, b_init
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b))

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost_history[-1]:.2f}, w={w:.2f}, b={b:.2f}")

    return w, b, cost_history

# Train
w_final, b_final, costs = gradient_descent(X, y, w_init=0, b_init=0, alpha=0.1, num_iters=1000)
print(f"\nFinal model: price = {w_final:.2f} * size + {b_final:.2f}")

# Predict a new house: 2800 sqft -> 2.8 (in our scaled units)
new_size = 2.8
predicted_price = w_final * new_size + b_final
print(f"Predicted price for 2800 sqft house: ${predicted_price:.0f}k")

# Visualize
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(X, y, color='red', label='Actual data')
plt.plot(X, w_final*X + b_final, color='blue', label='Fitted line')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price ($1000s)')
plt.legend()
plt.title('Linear Regression Fit')

plt.subplot(1,2,2)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration (should decrease)')

plt.tight_layout()
plt.savefig('linear_regression_scratch.png')
plt.show()
```

**Why build it from scratch first?** Because once you've coded gradient descent yourself, `sklearn.LinearRegression()` stops being a magic black box — you'll know exactly what it's doing underneath.

---

## 4. Same Thing, Using scikit-learn

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).reshape(-1, 1)  # sklearn wants 2D input
y = np.array([200, 300, 400, 480, 560, 620])

model = LinearRegression()
model.fit(X, y)

print(f"w (coefficient): {model.coef_[0]:.2f}")
print(f"b (intercept): {model.intercept_:.2f}")

prediction = model.predict([[2.8]])
print(f"Predicted price for 2800 sqft: ${prediction[0]:.0f}k")
```

Notice: same answer, 5 lines instead of 40. This is why you learn the math first, then use the library.

---

## 5. Multiple Linear Regression

Real houses depend on more than size — bedrooms, age, location, etc. Now `X` has multiple columns (features), and the model becomes:

```
f(x) = w1*x1 + w2*x2 + w3*x3 + ... + b
```

### Real-life example: California Housing dataset (built into sklearn, real data)

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load real data: 20,000+ California houses with 8 features
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print(df.columns.tolist())
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
#  'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']  # target: median house value ($100,000s)

# Split into train/test — NEVER evaluate on data the model trained on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling — crucial when features have very different ranges
# (MedInc is ~0-15, Population is 0-30000 — gradient descent struggles without scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # use SAME scaler, don't refit on test data

# Train
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")  # closer to 1.0 = better

# Which features matter most?
importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance (scaled coefficients):")
print(importance)
```

**Interpreting this:** `MedInc` (median income in the area) will almost always come out as the strongest predictor of house value — a real, intuitive insight, not just a number.

---

## 6. PROJECT 1: House Price Predictor (full pipeline)

Build this yourself, end to end. Steps:

1. Load the California housing dataset (above).
2. Explore it — plot histograms of each feature (`df.hist(figsize=(12,10))`).
3. Check for correlation with the target: `df.corr()['MedHouseVal'].sort_values()`.
4. Split into train/test.
5. Scale features.
6. Train `LinearRegression`.
7. Evaluate with R² and MSE.
8. **Bonus:** try dropping the weakest 2 features — does performance change?
9. **Bonus:** plot predicted vs actual values:

```python
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, 5], [0, 5], 'r--')  # perfect prediction line
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Predicted vs Actual House Prices')
plt.savefig('predictions_vs_actual.png')
plt.show()
```

A tight diagonal cluster = good model. A wide scatter = the model is missing something (which is normal for plain linear regression — later, boosted trees will do much better on this exact dataset, and you'll be able to compare).

---

## What's next

Phase 3 covers **Logistic Regression, Classification, Overfitting, and Regularization** — same structure: theory → from-scratch code → sklearn → real dataset → project (a spam/cancer classifier).

Say "continue to phase 3" when you're ready, or ask me anything about what's above first.
