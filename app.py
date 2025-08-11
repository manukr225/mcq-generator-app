import streamlit as st
from pytube import YouTube
import whisper
import tempfile
import os
import torch
from transformers import pipeline
import random

# Caching models
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

whisper_model = load_whisper_model()
summarizer = load_summarizer()

st.set_page_config(page_title="AI Quiz Generator", layout="centered")
st.title("📚 AI Quiz Generator (Free & Open Source)")

# --- Input Type Selection ---
input_type = st.radio("Select Input Type:", ["YouTube Link", "Paste Text"])

transcript = ""

if input_type == "YouTube Link":
    yt_url = st.text_input("Enter YouTube video URL:")

    if yt_url:
        try:
            with st.spinner("📥 Downloading and extracting audio..."):
                yt = YouTube(yt_url)
                stream = yt.streams.filter(only_audio=True).first()
                temp_audio = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                stream.download(filename=temp_audio.name)

            with st.spinner("🧠 Transcribing video using Whisper..."):
                result = whisper_model.transcribe(temp_audio.name)
                transcript = result['text']
                os.remove(temp_audio.name)

            st.success("✅ Transcription Complete!")
            st.text_area("📜 Transcript Preview:", value=transcript, height=200)

        except Exception as e:
            st.error(f"Error: {e}")

elif input_type == "Paste Text":
    transcript = st.text_area("Paste your content here:", height=300)

# --- Settings ---
if transcript:
    st.subheader("⚙️ Quiz Settings")
    num_questions = st.slider("Number of questions:", 1, 10, 3)

    if st.button("🧠 Generate Quiz"):
        with st.spinner("Summarizing content..."):
            chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
            summary = ""
            for chunk in chunks:
                summary += summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] + " "

        with st.spinner("🧪 Generating Questions..."):

            def generate_dummy_mcq(summary, num=3):
                # Very basic MCQ generator using summary keywords
                questions = []
                keywords = list(set(summary.split()))
                for i in range(num):
                    if len(keywords) < 4:
                        break
                    word = random.choice(keywords)
                    question = f"What is related to '{word}'?"
                    options = random.sample(keywords, 4)
                    answer = options[0]
                    explanation = f"The keyword '{word}' is associated with '{answer}' in the summary."
                    questions.append({
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "explanation": explanation
                    })
                return questions

            mcqs = generate_dummy_mcq(summary, num_questions)

        st.success("✅ Quiz Ready!")

        for i, mcq in enumerate(mcqs):
            st.markdown(f"**Q{i+1}: {mcq['question']}**")
            for opt in mcq["options"]:
                st.markdown(f"- {opt}")
            st.markdown(f"✅ **Answer**: {mcq['answer']}")
            with st.expander("💡 Explanation"):
                st.write(mcq["explanation"])

        st.markdown("---")
