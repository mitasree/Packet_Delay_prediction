# 📡 Packet Delay Prediction using Machine Learning  

## 📌 Overview  
This project focuses on predicting network packet delay using **machine learning models**.  
It was developed during my **Final Year Project (2024)** and uploaded here for reference.  

The goal was to design an **end-to-end ML pipeline** capable of handling heterogeneous data (string + integer parameters), preprocess it efficiently, and achieve high prediction accuracy for real-world network scenarios.  

---

## 🔬 Methodology  
- **Dataset**: Synthetic dataset generated to simulate realistic packet delay scenarios.  
- **Preprocessing**:  
  - One-Hot Encoding for categorical (string) parameters  
  - Normalization of numerical features  
  - Feature selection and correlation analysis for efficiency  
- **Pipeline**: Implemented an end-to-end Scikit-learn pipeline for reproducibility and modularity  
- **Models**: Regression models (Linear Regression, Random Forest, etc.) tested  
- **Frameworks**: Python, Scikit-learn, Pandas, NumPy  
- **Evaluation Metrics**: Accuracy and Mean Squared Error (MSE)  

---

## ⚙️ Features  
- Predicts **packet delay** given a set of network parameters  
- End-to-end automated **ML pipeline**: preprocessing → encoding → training → evaluation  
- Robust handling of **mixed-type data** (categorical + numerical)  
- Modular design, easy to extend with additional models  

---

## 📊 Results  
- **Accuracy**: 95%  
- **MSE**: 20  

These results demonstrate strong predictive performance and validate the effectiveness of the preprocessing and pipeline-based approach.  

---

## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/packet-delay-prediction.git
   cd packet-delay-prediction
