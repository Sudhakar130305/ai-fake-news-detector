AI & Fake News Detector
An intelligent web application that detects whether a news article is AI-generated or human-written and verifies its authenticity using the Google Fact-Check API. Built using DistilBERT, Streamlit, and Transformers.

LIVE LINK https://ai-fake-news-detector-9jxuk6kb6w36fb2dhiwgfq.streamlit.app/

Features

 Detects AI-generated vs Human-written text
 Displays AI confidence score
 Verifies news using Google Fact-Check API
 User-friendly Streamlit web interface
 Fast and accurate real-time analysis

Model Used

DistilBERT for Sequence Classification
Fine-tuned on a labeled AI vs Human news dataset
Optimized using the HuggingFace Transformers Trainer

Tech Stack

Python
Streamlit
HuggingFace Transformers
PyTorch
Google Fact-Check API

How to Run Locally
# 1️⃣ Clone the repository
git clone https://github.com/Sudhakar130305/ai-fake-news-detector.git

# 2️⃣ Navigate into the project
cd ai-fake-news-detector

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
streamlit run app.py

Use Cases

Fake news detection
AI content verification
Journalism & media trust validation
Academic research
Social media content moderation
