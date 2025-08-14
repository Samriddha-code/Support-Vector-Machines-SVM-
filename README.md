Support Vector Machines (SVM)

## 📌 Objective
The goal of this task is to build and evaluate **Support Vector Machine (SVM)** models for the binary classification of breast tumors (**Malignant vs Benign**) using the provided `breast-cancer.csv` dataset.  
We implement both **Linear** and **RBF** kernels, compare their performance, optimize using hyperparameter tuning, and visualize decision boundaries on synthetic data.

---

## 📊 Dataset
- **File:** `breast-cancer.csv` (provided in task resources)
- **Description:** Diagnostic data of breast tumors, containing several measured features of cell nuclei.
- **Target column:** `diagnosis`
  - `M` → **1** (Malignant)
  - `B` → **0** (Benign)
- **Total samples:** 569  
- **Features:** 30 numeric features (excluding `id` and `diagnosis`)  
- **Class distribution:**  
  - Malignant: 212 samples  
  - Benign: 357 samples  

---

---

## 🚀 Steps Implemented

### 1️⃣ Data Loading & Preprocessing
- Loaded the CSV file using **Pandas**.
- Encoded `diagnosis` values:
  - Malignant (`M`) → `1`
  - Benign (`B`) → `0`
- Dropped the unnecessary `id` column.
- Split into **Train (80%)** and **Test (20%)** datasets using `train_test_split`.
- Scaled features using **StandardScaler** for better SVM performance.

### 2️⃣ Model Training
- **Linear SVM** → `SVC(kernel='linear')`
- **RBF SVM** → `SVC(kernel='rbf')`
- Trained both models on the scaled training data.

### 3️⃣ Model Evaluation
- Calculated **Test Accuracy** for both kernels.
- Performed **5-fold Cross-Validation** for more robust evaluation.

### 4️⃣ Hyperparameter Tuning
- Used `GridSearchCV` to optimize RBF SVM.
- Tuned:
  - **C:** `[0.1, 1, 10, 100]`
  - **gamma:** `['scale', 'auto', 0.001, 0.01, 0.1, 1]`
- Selected the best parameter combination for maximum CV score.

### 5️⃣ Visualization (2D Example)
- Created a **synthetic 2D dataset** with `make_classification`.
- Plotted decision boundaries for:
  - Linear kernel
  - RBF kernel
- Highlighted **support vectors** in the plots.

---

## 📈 Sample Results

| Model              | Test Accuracy | CV Mean Score |
|--------------------|--------------|---------------|
| Linear SVM         | 0.9649       | 0.9753        |
| RBF SVM            | 0.9561       | 0.9684        |
| Optimized RBF SVM  | 0.9737       | 0.9800        |

> 📌 *Numbers may vary slightly due to random data splits.*

---
