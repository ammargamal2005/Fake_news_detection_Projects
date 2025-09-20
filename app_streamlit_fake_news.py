import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ================== Load Model and Tokenizer ==================
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(r"D:\Ammar\Ai Diploma\projects\fake news detection\rnn_lstm_model.h5")
    with open(r"D:\Ammar\Ai Diploma\projects\fake news detection\tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

MAX_LEN = 1000

# ================== Streamlit Config ==================
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# Sidebar
st.sidebar.title("📰 Fake News Detection")
page = st.sidebar.radio("Navigate", ["📖 Project Description", "🔍 Prediction", "📊 Evaluation"])

# ================== Project Description Page ==================
if page == "📖 Project Description":
    st.title("📰 Fake News Detection using RNN/LSTM")
    st.markdown(
        """
        ## Project Overview  
        Fake news has become one of the biggest challenges in modern society, especially with the rise of social media.  
        This project aims to **detect fake vs. real news** using a **Recurrent Neural Network (RNN/LSTM)** model.  

        ### Workflow:
        1. **Data Preparation**  
           - Collected data from `true.csv` and `fake.csv` (2015–2018).  
           - Added a label column (`0 = Real`, `1 = Fake`).  
           - Cleaned duplicates and formatted dates.  
           - Text preprocessing (lowercasing, punctuation removal, numbers removal, stemming).  

        2. **Feature Engineering & Model Building**  
           - Balanced dataset using **RandomOverSampler**.  
           - Text converted to sequences using **Tokenizer + pad_sequences (maxlen=1000)**.  
           - Model Architecture:  
             - **Embedding Layer**  
             - **LSTM Layer**  
             - **Dense + Dropout**  
           - Compiled with **Adam optimizer, binary_crossentropy loss, metrics = Accuracy & AUC**.  

        3. **Training & Evaluation**  
           - **EarlyStopping** to prevent overfitting.  
           - Trained for 4 epochs.  
           - Saved model & tokenizer for deployment.  

        ### Example Prediction
        Input:  
        `"A major tech company announced a new smartphone that can fully charge ... using air power technology ... for only $50"`  

        Prediction:  
        - Probability: **0.99999666**  
        - Classified as: **Fake (1)**  

        ✅ The model successfully identifies unrealistic news as Fake.
        ---
        """
    )

    st.image(
        "https://images.theconversation.com/files/284418/original/file-20190717-173334-1b9vdud.jpg?ixlib=rb-4.1.0&rect=0%2C0%2C6490%2C3957&q=20&auto=format&w=320&fit=clip&dpr=2&usm=12&cs=strip",
        use_container_width=True
    )
    st.success("This is just the description page. Switch to 'Prediction' or 'Evaluation' to test the model!")

# ================== Prediction Page ==================
elif page == "🔍 Prediction":
    st.title("🔍 Fake News Prediction")

    # Inputs
    title_input = st.text_input("📝 Enter News Title", "")
    text_input = st.text_area("📜 Enter News Content", "")

    if st.button("Predict"):
        if title_input.strip() == "" or text_input.strip() == "":
            st.warning("⚠️ Please enter both Title and Content.")
        else:
            # Concatenate title + text
            final_text = title_input + " " + text_input

            # Tokenize + Pad
            seq = tokenizer.texts_to_sequences([final_text])
            padded = pad_sequences(seq, maxlen=MAX_LEN)

            # Prediction
            pred = model.predict(padded)[0][0]
            percentage = round(float(pred) * 100, 2)

            # Result
            if pred >= 0.5:
                st.error(f"🚨 This news is **Fake** with probability {percentage}%")
                st.image("https://storage.googleapis.com/kaggle-datasets-images/7266777/11589020/9692fbe0ea1b4822642af9ddb863643d/dataset-cover.jpg?t=2025-04-27-14-59-53", width=150)
            else:
                st.success(f"✅ This news is **Real** with probability {100 - percentage}%")
                st.image("https://bucket.zammit.shop/active-storage/8zbwc6g7a0zabdedjkgxabe04bmw", width=150)

# ================== Evaluation Page ==================
# ================== Evaluation Page ==================
elif page == "📊 Evaluation":
    st.title("📊 Model Evaluation")

    st.markdown(
        """
        This page shows the **evaluation results** of the Fake News Detection model.  
        - Confusion Matrix (uploaded as an image)  
        - You can also add other plots in the future (ROC Curve, Precision-Recall, etc.)  
        """
    )

    # 🔹 Show Confusion Matrix image (replace with your file path)
    st.subheader("Confusion Matrix")
    st.image(r"D:\Ammar\Ai Diploma\projects\fake news detection\output.png", caption="Confusion Matrix", use_container_width=True)

    # 🔹 Placeholder for another subplot (example image)
    st.subheader("Additional Metric ")
    st.image(r"D:\Ammar\Ai Diploma\projects\fake news detection\output1.png", caption="ROC Curve Example", use_container_width=True)


# ================== Footer ==================
st.markdown(
    """
    ---
    <div style="text-align: center; font-size: 16px;">
        Made with ❤️ by <b>Eng. Ammar Gamal</b> <br>
        🌐 <a href="https://www.ammar-gamal.me/" target="_blank">www.ammar-gamal.me</a>
    </div>
    """,
    unsafe_allow_html=True
)
