# 🧠 MNIST Digit & Fashion-MNIST Classification Project

This project focuses on implementing linear classifiers from scratch to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and also developing dense and convolutional neural networks for the [Fashion-MNIST dataset](https://keras.io/api/datasets/fashion_mnist/). The goal is to understand both foundational linear models and practical deep learning architectures using Jupyter Notebooks.

---

## 📦 Project Structure

```
mnist_project/
├── data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── utils/
│   ├── feature_extraction.py
│   ├── hinge_loss.py
│   ├── perceptron.py
│   └── pegasos.py
├── classifiers/
│   ├── perceptron_classifier.py
│   ├── pegasos_classifier.py
├── notebooks/
│   ├── MNIST_Linear_Models.ipynb
│   ├── FashionMNIST_DenseNN.ipynb
│   └── FashionMNIST_CNN.ipynb
├── main.py
└── README.md
```

---

## ✅ Objectives

- Preprocess the MNIST and Fashion-MNIST datasets
- Implement and evaluate:
  - Perceptron, Average Perceptron, Pegasos (SVM with hinge loss)
  - Dense Neural Network (DNN) for Fashion-MNIST
  - Convolutional Neural Network (CNN) for Fashion-MNIST
- Perform hyperparameter tuning
- Evaluate models using accuracy and loss metrics

---

## 🔧 Implemented Algorithms

### Linear Classifiers (from scratch)

#### 1. Hinge Loss
```python
def hinge_loss_single(feature_vector, label, theta, theta_0):
    return max(0, 1 - label * (np.dot(feature_vector, theta) + theta_0))
```

#### 2. Perceptron
```python
def perceptron_single_step_update(x, y, theta, theta_0):
    if y * (np.dot(theta, x) + theta_0) <= 0:
        theta += y * x
        theta_0 += y
    return theta, theta_0
```

#### 3. Average Perceptron
```python
def average_perceptron(feature_matrix, labels, T):
    # Averages the weight and bias vectors across all updates
```

#### 4. Pegasos (SVM)
```python
def pegasos_single_step_update(x, y, L, eta, theta, theta_0):
    # Stochastic gradient descent for SVM
```

---

### Deep Learning Models (Keras + Jupyter)

#### Dense Neural Network (DNN)

- Two hidden layers: [512, 256]
- Optimizers: `rmsprop`, `adam`
- Validation-based epoch selection
- Hyperparameter tuning: batch size, layers, optimizer

#### Convolutional Neural Network (CNN)

- Two convolution layers: [32, 64] with 3x3 kernels
- Max pooling: 2x2
- Dense layer with 64 neurons before output
- Hyperparameter tuning: channel size, batch size, optimizer

---

## 📊 Evaluation

- **Accuracy** for classification performance
- **Hinge Loss** for linear models
- **Cross-validation** for model selection
- **Model.summary()** to compare number of parameters

---

## 🚀 Running the Project

### For MNIST Linear Classifiers
```bash
python main.py --classifier perceptron --T 10
python main.py --classifier pegasos --T 25 --lambda 0.01
```

### For Fashion-MNIST Deep Learning Models
Use Jupyter Notebooks:
- `FashionMNIST_DenseNN.ipynb`
- `FashionMNIST_CNN.ipynb`

---

## 📈 Results Snapshot

| Classifier         | Dataset        | Accuracy (Test) |
|--------------------|----------------|-----------------|
| Perceptron         | MNIST          | 88.5%           |
| Avg. Perceptron    | MNIST          | 89.6%           |
| Pegasos (λ=0.01)   | MNIST          | 90.2%           |
| Dense NN (Tuned)   | Fashion-MNIST  | TBD             |
| CNN (Tuned)        | Fashion-MNIST  | TBD             |

---

## 🧪 Future Work

- Add Logistic Regression and Softmax classifiers
- Explore kernelized methods for linear models
- Visualize filters and feature maps in CNN
- Automate hyperparameter tuning (e.g., using GridSearch or Optuna)

---

## 📚 References

- [Deep Learning with Python – Chollet](https://www.manning.com/books/deep-learning-with-python)
- [Scikit-learn SVM Docs](https://scikit-learn.org/stable/modules/svm.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST Dataset](https://keras.io/api/datasets/fashion_mnist/)
