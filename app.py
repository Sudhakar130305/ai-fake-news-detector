import torch
import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import requests

# -----------------------------
# Load Model
# -----------------------------
MODEL_DIR = "final_model"  # Local model folder

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# -----------------------------
# Fact Check API
# -----------------------------
FACTCHECK_API_KEY = "YOUR_API_KEY_HERE"   # Replace

def google_fact_check(query):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
    "query": query,
    "key": "AIzaSyCClCVtgBGBMAH_Fi0GzQzVSE3kYPxv70E",
    "pageSize": 5
}


    try:
        r = requests.get(url, params=params)
        return r.json()
    except:
        return {"error": "API error"}

# -----------------------------
# AI Probability Rescale
# -----------------------------
def rescale_probability(prob, threshold=0.9998):
    if prob >= threshold:
        return 0.6 + (prob - threshold) * (0.4 / (1 - threshold))
    else:
        return (prob / threshold) * 0.4

# -----------------------------
# Prediction Function
# -----------------------------
def analyze(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    raw_prob = outputs.logits.softmax(dim=1)[0][1].item()
    scaled = rescale_probability(raw_prob)
    prediction = "AI Generated" if raw_prob >= 0.9998 else "Human Written"

    fc = google_fact_check(text[:120])
    
    if "claims" in fc:
        claim = fc["claims"][0]["text"]
        rating = fc["claims"][0]["claimReview"][0]["textualRating"]
        source = fc["claims"][0]["claimReview"][0]["publisher"]["name"]
        fact_result = f"{rating}  |  Source: {source}"
    else:
        fact_result = "No fact-check results found."

    return prediction, scaled, fact_result

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📰 AI + Fake News Detector")
st.write("Enter a news paragraph below to check whether it's AI-generated and verify facts.")

text = st.text_area("Enter news text:", height=200)

if st.button("Analyze"):
    pred, score, fact = analyze(text)
    
    st.subheader("🔍 AI Detection")
    st.write(f"Prediction: **{pred}**")
    st.write(f"Confidence Score: **{score * 100:.2f}%**")

    st.subheader("📡 Fact Check")
    st.write(fact)




