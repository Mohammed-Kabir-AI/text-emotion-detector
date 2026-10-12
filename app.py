import re
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Text Emotion Detector", layout="wide")

MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOJI = {"anger": "😠", "disgust": "🤢", "fear": "😨", "joy": "😀",
         "neutral": "😐", "sadness": "😢", "surprise": "😲"}
ORDER = list(EMOJI)

@st.cache_resource
def load_model():
    return pipeline("text-classification", model=MODEL, top_k=None)

clf = load_model()

def predict(texts):
    out = clf(texts, truncation=True, max_length=512, batch_size=16)
    if out and isinstance(out[0], dict):
        out = [out]
    rows = [{d["label"]: d["score"] for d in res} for res in out]
    return pd.DataFrame(rows)[ORDER]

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]

st.title("🎭 Text Emotion Detector")
st.caption("Educational project. English text only. Predictions are estimates and can be wrong. "
           "Text is processed in memory and never stored.")

tab1, tab2, tab3, tab4 = st.tabs(["Single Text", "Emotional Arc", "Batch CSV", "About"])

with tab1:
    text = st.text_area("Enter English text", "I can't believe I finally got the job!", height=120)
    if st.button("Detect emotion", key="single"):
        if len(text.strip()) < 3:
            st.warning("Please enter some text.")
        else:
            scores = predict([text]).iloc[0]
            top = scores.idxmax()
            st.subheader(f"{EMOJI[top]} {top.title()} ({scores[top]*100:.1f}%)")
            if scores[top] < 0.5:
                st.info("The model is not confident. The text may be mixed, neutral or ambiguous.")
            st.bar_chart(scores)

with tab2:
    st.write("Paste a story, speech or video script to see how emotions change over time.")
    long_text = st.text_area("Long text or script", height=250, key="arc")
    if st.button("Analyze emotional arc"):
        sentences = split_sentences(long_text)[:200]
        if len(sentences) < 3:
            st.warning("Please paste at least 3 sentences.")
        else:
            with st.spinner("Analyzing..."):
                df = predict(sentences)
            df.index = range(1, len(df) + 1)
            st.subheader("Emotion over the text (sentence by sentence)")
            st.line_chart(df)
            st.subheader("Overall emotion mix")
            st.bar_chart(df.mean())
            table = pd.DataFrame({
                "Sentence": sentences,
                "Top emotion": [f"{EMOJI[e]} {e}" for e in df.idxmax(axis=1)],
                "Confidence": (df.max(axis=1) * 100).round(1),
            })
            st.dataframe(table, use_container_width=True)

with tab3:
    up = st.file_uploader("Upload a CSV file", type=["csv"])
    if up:
        data = pd.read_csv(up)
        col = st.selectbox("Which column has the text?", data.columns)
        if st.button("Run batch analysis"):
            subset = data.dropna(subset=[col]).head(500).copy()
            with st.spinner(f"Analyzing {len(subset)} rows..."):
                res = predict(subset[col].astype(str).tolist())
            subset["emotion"] = res.idxmax(axis=1).values
            subset["confidence"] = (res.max(axis=1) * 100).round(1).values
            st.bar_chart(subset["emotion"].value_counts())
            st.dataframe(subset, use_container_width=True)
            st.download_button("Download results (CSV)",
                               subset.to_csv(index=False), file_name="emotion_results.csv")
        st.caption("Only the first 500 rows are analyzed to stay within free-tier limits.")

with tab4:
    st.markdown("""
**Model:** j-hartmann/emotion-english-distilroberta-base (Hugging Face).  
**Emotions:** anger, disgust, fear, joy, neutral, sadness, surprise.  
**Credit:** Jochen Hartmann, "Emotion English DistilRoBERTa-base", 2022.

**Limitations**
- English only
- Struggles with sarcasm, slang, mixed emotions and very short text
- About 66% accuracy on 7 classes, so treat results as estimates
- Not a tool for diagnosing anyone's mental state
""")
