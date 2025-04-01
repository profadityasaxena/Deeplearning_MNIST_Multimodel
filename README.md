# 🧠 MNIST & Fashion-MNIST Classification Project

#### [Author] : Aditya Saxena

This project explores both classic and modern techniques for image classification using two well-known datasets:

- **MNIST**: 70,000 grayscale images of handwritten digits (0–9)
- **Fashion-MNIST**: 70,000 grayscale images of fashion items (10 categories)

---

## 🎯 Objectives

- Implement linear classifiers from scratch:  
  - Perceptron  
  - Average Perceptron  
  - Pegasos (SVM with hinge loss)

- Implement deep learning models using Keras:  
  - Dense Neural Networks (DNN)  
  - Convolutional Neural Networks (CNN)

- Evaluate models using:  
  - Accuracy  
  - Hinge loss  
  - Zero-One loss  
  - Confusion matrix

- Tune hyperparameters (e.g., learning rate, epochs, batch size)
- Visualize metrics (accuracy/loss curves, decision boundaries)

---

## 🗂️ Project Structure

```
DeepLearning_MNIST_Multimodel/
└── Project/
    ├── DeepLearning_MNIST_LinearModel.ipynb                # Linear classifiers: Perceptron, Pegasos, Avg. Perceptron
    ├── DeepLearning_MNIST_DeepLearningAdvancedModel.ipynb  # DNN and CNN models using Keras
    └── README.md                                           # Project documentation
```

---

## 📦 Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Keras + TensorFlow backend
- Jupyter Notebook

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## 🧪 Results Summary

| Model               | Dataset        | Train Accuracy | Test Accuracy |
|---------------------|----------------|----------------|----------------|
| Perceptron          | MNIST (Binary) | 91.0%          | 87.6%          |
| Avg. Perceptron     | MNIST (Binary) | 92.5%          | 89.2%          |
| Pegasos (λ=0.01)    | MNIST (Binary) | 94.1%          | 90.5%          |
| DNN (Tuned)         | Fashion-MNIST  | 96.8%          | 89.3%          |
| CNN (Tuned)         | Fashion-MNIST  | 98.6%          | 91.2%          |

---

## 📈 Visualizations

- Accuracy/loss over epochs
- Confusion matrices
- Sample predictions and misclassifications
- Decision boundaries (for 2D toy datasets)

---

## 🔧 Utility Functions

- `classify(...)`: Predict class using linear rule  
- `classification_accuracy(...)`: Measure accuracy  
- `hinge_loss_single(...)` and `hinge_loss_full(...)`: Compute hinge loss  
- `zero_one_loss(...)`: Compute general classification error  
- `plot_metrics_over_epochs(...)`: Visualize accuracy/loss over time  
- `report_results_table(...)`: Display results as formatted table  

---

## 🧠 Learning Outcomes

- Hands-on implementation of linear classifiers  
- Practical comparison with deep learning models  
- Visual intuition on margins and loss  
- Integration of model performance evaluation + tuning

---

## 📚 References

- Yann LeCun's MNIST dataset: http://yann.lecun.com/exdb/mnist/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Deep Learning with Python – François Chollet
