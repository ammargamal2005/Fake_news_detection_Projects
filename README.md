# 📰 Fake News Detection using RNN/LSTM

This project implements a **Fake News Detection system** using a **Recurrent Neural Network (RNN/LSTM)**. The app is built with **Streamlit** and allows users to predict whether a news article is real or fake.

---

## 📌 Project Overview

Fake news has become a major challenge in modern society, especially with the rise of social media.
This project aims to **detect fake vs. real news** using machine learning and deep learning techniques.

**Workflow:**

1. **Data Preparation**

   * Collected data from `true.csv` and `fake.csv` (2015–2018).
   * Added a label column (`0 = Real`, `1 = Fake`).
   * Cleaned duplicates and formatted dates.
   * Text preprocessing: lowercasing, punctuation removal, numbers removal, stemming.

2. **Feature Engineering & Model Building**

   * Balanced dataset using **RandomOverSampler**.
   * Converted text to sequences using **Tokenizer** and padded sequences (`maxlen=1000`).
   * Model Architecture:

     * **Embedding Layer**
     * **LSTM Layer**
     * **Dense + Dropout Layers**
   * Compiled with **Adam optimizer**, **binary\_crossentropy loss**, and **accuracy & AUC metrics**.

3. **Training & Evaluation**

   * Used **EarlyStopping** to prevent overfitting.
   * Trained for 4 epochs.
   * Saved the model (`rnn_lstm_model.h5`) and tokenizer (`tokenizer.pkl`) for deployment.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd <repo-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have:

* `streamlit`
* `tensorflow`
* `scikit-learn`
* `numpy`
* `pillow`
* `matplotlib`

---

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### Pages in the App:

1. **Project Description**

   * Overview of the project and workflow.
   * Example input and prediction.
2. **Prediction**

   * Enter news title and content.
   * Click **Predict** to see if the news is Real or Fake.
3. **Evaluation**

   * View Confusion Matrix and other evaluation metrics.
   * Placeholder for ROC Curve and future metrics.

---

## 📊 Example Prediction

**Input:**
"A major tech company announced a new smartphone that can fully charge ... using air power technology ... for only \$50"

**Output:**

* Probability: **0.99999666**
* Classified as: **Fake (1)**

---

## 🖼 Images

* Project Overview:
  ![Project Image](https://images.theconversation.com/files/284418/original/file-20190717-173334-1b9vdud.jpg?ixlib=rb-4.1.0\&rect=0%2C0%2C6490%2C3957\&q=20\&auto=format\&w=320\&fit=clip\&dpr=2\&usm=12\&cs=strip)

* Prediction Example:

  * Fake news image:
    ![Fake Image](https://storage.googleapis.com/kaggle-datasets-images/7266777/11589020/9692fbe0ea1b4822642af9ddb863643d/dataset-cover.jpg?t=2025-04-27-14-59-53)
  * Real news image:
    ![Real Image](https://bucket.zammit.shop/active-storage/8zbwc6g7a0zabdedjkgxabe04bmw)

* Evaluation:

  * Confusion Matrix: `output.png`
  * ROC Curve Example: `output1.png`

---

## 📝 Footer

Made with ❤️ by **Eng. Ammar Gamal**
🌐 [www.ammar-gamal.me](https://www.ammar-gamal.me/)
