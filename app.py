
import streamlit as st
from newspaper import Article
import pickle

# Load model and supporting objects
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("bias_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

def extract_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def predict_bias(text):
    vector = vectorizer.transform([text])
    proba = model.predict_proba(vector)[0]
    return {label_encoder.classes_[i]: round(p * 100, 2) for i, p in enumerate(proba)}

# Streamlit UI
st.title("Political Bias Analyzer")
st.write("Paste a news article URL and find out the estimated political leaning by percentage.")

url = st.text_input("Enter Article URL")

if st.button("Analyze"):
    if url:
        with st.spinner("Reading and analyzing article..."):
            try:
                text = extract_text(url)
                result = predict_bias(text)
                st.success("Analysis Complete!")
                st.subheader("Bias Prediction:")
                st.write(result)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid URL.")
