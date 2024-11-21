# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from keras.models import load_model
import pickle

# Load the model and tokenizer
@st.cache_resource
def load_lstm_model():
    model = load_model('TestModelLSTMFix.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_lstm_model()

# Helper functions
def preprocess_input(text, tokenizer, max_length=10):
    """Preprocess user input text for model prediction."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')
    return np.array(padded_sequence).reshape(1, 1, -1)

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a single input text."""
    processed_text = preprocess_input(text, tokenizer)
    prediction = model.predict(processed_text)
    sentiment = "Positif" if prediction > 0.5 else "Negatif"
    return sentiment, prediction

def display_confusion_matrix(y_true, y_pred):
    """Display confusion matrix using Streamlit."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'],
                yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

# Streamlit UI
st.title("Sentimen Analisis Berbasis Aspek")
st.markdown("""
Aplikasi ini memprediksi sentimen kalimat (Positif, Negatif, Netral) dan menganalisis aspek tertentu 
seperti **performance**, **user interface**, dan **fitur Gojek**.
""")

# User input
user_text = st.text_area("Masukkan kalimat ulasan Anda:", placeholder="Ketik ulasan di sini...")
selected_aspect = st.selectbox("Pilih aspek analisis sentimen:", ["Performance", "User Interface", "Fitur Gojek"])

# Prediction
if st.button("Prediksi Sentimen"):
    if user_text.strip():
        sentiment, prediction = predict_sentiment(user_text, model, tokenizer)
        st.success(f"Sentimen untuk aspek **{selected_aspect}** adalah **{sentiment}**.")
        st.info(f"Confidence Score: {prediction[0][0]:.2f}")
    else:
        st.error("Masukkan kalimat ulasan terlebih dahulu!")

# Upload dataset for evaluation
st.markdown("## Evaluasi Model")
uploaded_file = st.file_uploader("Unggah file CSV dengan kolom ulasan dan label sentimen:", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'content' in data.columns and 'sentiment' in data.columns:
        st.write("Data berhasil diunggah:")
        st.dataframe(data.head())

        # Preprocess data
        data['prediction'] = data['finalpreprocess'].apply(lambda x: predict_sentiment(x, model, tokenizer)[0])

        # Calculate metrics
        y_true = data['sentiment'].map({'Negatif': 0, 'Netral': 1, 'Positif': 2}).values
        y_pred = data['prediction'].map({'Negatif': 0, 'Netral': 1, 'Positif': 2}).values
        accuracy = (y_true == y_pred).mean()

        st.write(f"**Akurasi Model:** {accuracy:.2f}")
        display_confusion_matrix(y_true, y_pred)
    else:
        st.error("File CSV harus memiliki kolom 'content' dan 'sentiment'.")