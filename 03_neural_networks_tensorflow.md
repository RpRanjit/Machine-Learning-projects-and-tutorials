# Machine Learning Course — Phase 4
## Neural Networks, TensorFlow & Multiclass Classification

---

## 1. From Logistic Regression to Neural Networks

Logistic regression is basically **one neuron**: `sigmoid(w·x + b)`.

A **neural network** stacks many of these neurons in **layers**, letting the model learn far more complex patterns than a straight line or single curve.

```
Input layer  ->  Hidden layer(s)  ->  Output layer
  (features)      (learned features)    (prediction)
```

**Real-life intuition:** to recognize a handwritten digit, you don't look at raw pixels directly — your brain first detects edges, then shapes (loops, lines), then combines those into "this is a 8". Each hidden layer in a neural network learns increasingly abstract features the same way, automatically, from data.

---

## 2. Anatomy of a Neuron

Each neuron does exactly what logistic regression does:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
a = g(z)     <- g is the "activation function"
```

### Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): return 1/(1+np.exp(-z))
def relu(z): return np.maximum(0, z)
def tanh(z): return np.tanh(z)

z = np.linspace(-5, 5, 100)

plt.figure(figsize=(12,3))
for i, (name, fn) in enumerate([('Sigmoid', sigmoid), ('ReLU', relu), ('Tanh', tanh)]):
    plt.subplot(1,3,i+1)
    plt.plot(z, fn(z))
    plt.title(name)
    plt.grid(True)
plt.tight_layout()
plt.savefig('activations.png')
plt.show()
```

- **ReLU** (`max(0,z)`) — the default choice for hidden layers today. Fast, avoids "vanishing gradient" problems.
- **Sigmoid** — used in the OUTPUT layer for binary classification (gives a 0-1 probability).
- **Softmax** — used in the output layer for MULTIclass classification (gives probabilities across many classes that sum to 1).

---

## 3. Forward Propagation — By Hand (understand it before TensorFlow does it for you)

```python
import numpy as np

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1/(1+np.exp(-z))

# A tiny network: 2 inputs -> 3 hidden neurons -> 1 output
np.random.seed(1)
X = np.array([0.5, 0.8])   # one example, 2 features

# Layer 1 weights: 3 neurons, each with 2 input weights + 1 bias
W1 = np.random.randn(3, 2) * 0.1
b1 = np.zeros(3)

# Layer 2 (output) weights: 1 neuron, 3 input weights + 1 bias
W2 = np.random.randn(1, 3) * 0.1
b2 = np.zeros(1)

# Forward pass
z1 = np.dot(W1, X) + b1
a1 = relu(z1)                 # hidden layer output (3 numbers)
print("Hidden layer activations:", a1)

z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)              # final prediction
print("Output prediction:", a2)
```

This IS what "deep learning" means at its core — a chain of `(multiply, add, activate)` steps. Training just means using gradient descent (via **backpropagation**, the chain rule applied layer by layer) to find the `W1, b1, W2, b2` that make correct predictions.

---

## 4. Your First Neural Network with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reuse the cancer dataset from Phase 3
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model — this is the modern equivalent of what we hand-coded above
model = keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Dense(16, activation='relu'),   # hidden layer 1
    keras.layers.Dense(8, activation='relu'),    # hidden layer 2
    keras.layers.Dense(1, activation='sigmoid')  # output layer (binary)
])

model.compile(
    optimizer='adam',              # a smarter version of gradient descent
    loss='binary_crossentropy',    # same log loss from Phase 3
    metrics=['accuracy']
)

model.summary()  # prints the architecture and parameter count

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# Plot training curves — the single most useful debugging tool in deep learning
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title('Accuracy')
plt.savefig('nn_training_curves.png')
plt.show()
```

**Reading the training curves:** if `val_loss` starts climbing while `train_loss` keeps dropping, that's overfitting happening in real time — you'd add dropout, regularization, or stop training earlier (early stopping).

---

## 5. Multiclass Classification with Softmax

Binary → sigmoid → one number. Multiclass (e.g. digits 0-9, or types of iris flowers) → **softmax** → a probability for EACH class, all summing to 1.

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)   for all classes j
```

```python
import numpy as np

def softmax(z):
    e = np.exp(z - np.max(z))  # subtract max for numerical stability
    return e / np.sum(e)

# Example: raw scores for 3 classes (cat, dog, bird)
scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print(probs)              # e.g. [0.66, 0.24, 0.10] -- sums to 1.0
print(f"Predicted class: {np.argmax(probs)}")  # 0 = cat
```

Loss function used: **categorical cross-entropy** (the multiclass generalization of log loss).

---

## 6. PROJECT 3: Handwritten Digit Recognition (MNIST) — Real Deep Learning

This is the "hello world" of deep learning, using 70,000 real handwritten digit images.

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the data — comes built into Keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f"Training images: {X_train.shape}")  # (60000, 28, 28)
print(f"Test images: {X_test.shape}")       # (10000, 28, 28)

# Look at some examples
plt.figure(figsize=(10,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.savefig('mnist_samples.png')
plt.show()

# Preprocess: scale pixel values 0-255 to 0-1, flatten 28x28 image to 784 numbers
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

# Build a multiclass network: 784 inputs -> hidden layers -> 10 outputs (digits 0-9)
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),           # randomly "turns off" 20% of neurons each step -> reduces overfitting
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')   # 10 classes, probabilities sum to 1
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # use this when labels are plain integers (0-9), not one-hot
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")   # you should get ~97-98%

# Make predictions on real test images and visualize
predictions = model.predict(X_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    correct = predicted_labels[i] == y_test[i]
    color = 'green' if correct else 'red'
    plt.title(f"Pred: {predicted_labels[i]} (True: {y_test[i]})", color=color)
    plt.axis('off')
plt.tight_layout()
plt.savefig('mnist_predictions.png')
plt.show()

# Confusion matrix — where does the model get confused? (e.g. 4 vs 9, 3 vs 8)
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_all = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred_all)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — MNIST Digit Classification')
plt.savefig('mnist_confusion_matrix.png')
plt.show()
```

**Extend this project yourself:**
1. Try removing the `Dropout` layer — does the model overfit more (bigger gap between train/val accuracy)?
2. Try adding a third hidden layer — does accuracy improve, plateau, or get worse?
3. Try a smaller learning rate: `keras.optimizers.Adam(learning_rate=0.0001)` — train longer, compare.
4. Find 5 images the model got wrong. Do they look ambiguous even to you?

---

## 7. Model Optimization Basics

- **Learning rate:** too high → loss bounces around/explodes. Too low → painfully slow training. Use `learning_rate=0.001` as a solid default with Adam.
- **Batch size:** how many examples the model looks at before each weight update. Common values: 32, 64, 128. Smaller = noisier but sometimes generalizes better; larger = faster but needs more memory.
- **Epochs:** one full pass through the training data. Watch validation loss — stop when it stops improving (early stopping):

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,           # stop if no improvement for 5 epochs
    restore_best_weights=True
)

model.fit(X_train, y_train, validation_split=0.1, epochs=100,
          callbacks=[early_stop])
```

---

## What's next

**Phase 5** covers **Decision Trees, Random Forests, and Boosted Trees (XGBoost)** — a completely different, often more powerful approach for tabular/structured data, plus how to choose between neural nets and tree models for a given problem.

Say **"continue to phase 5"** when ready.
