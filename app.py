import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Title
# -----------------------------
st.title("📊 Financial Sentiment Analysis Dashboard")
st.write("Analyze financial tweets using BERT model")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert_model")
    tokenizer = BertTokenizer.from_pretrained("bert_model")
    return model, tokenizer

model, tokenizer = load_model()

# -----------------------------
# Label Mapping
# -----------------------------
labels = {
    0: "Bearish 📉",
    1: "Bullish 📈",
    2: "Neutral 😐"
}

# -----------------------------
# Input
# -----------------------------
user_input = st.text_area("Enter a financial tweet:")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader("Prediction:")
        st.success(labels[prediction])

        st.subheader("Confidence Scores:")
        prob_values = {
            "Bearish": float(probs[0][0]),
            "Bullish": float(probs[0][1]),
            "Neutral": float(probs[0][2])
        }
        st.write(prob_values)

        # -----------------------------
        # Bar Chart (Probabilities)
        # -----------------------------
        st.subheader("📊 Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(prob_values.keys(), prob_values.values())
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

# -----------------------------
# Dataset Distribution
# -----------------------------
st.subheader("📈 Dataset Class Distribution")

@st.cache_data
def load_data():
    df = pd.read_csv("sent_train.csv")
    return df

df = load_data()

class_counts = df["label"].value_counts().sort_index()

label_names = ["Bearish", "Bullish", "Neutral"]

fig2, ax2 = plt.subplots()
ax2.bar(label_names, class_counts.values)
ax2.set_title("Class Distribution in Dataset")
ax2.set_ylabel("Count")

st.pyplot(fig2)