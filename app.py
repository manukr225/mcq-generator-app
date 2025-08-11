import streamlit as st
from pytube import YouTube
import tempfile
import os
import openai
from transformers import pipeline

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Title
st.title("🎓 YouTube Video to MCQ Quiz Generator")

# Input YouTube link
yt_link = st.text_input("Paste YouTube video link")

# Start process
if yt_link:
    try:
        st.info("Downloading audio from YouTube...")
        yt = YouTube(yt_link)
        audio_stream = yt.streams.filter(only_audio=True).first()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_audio:
            audio_path = temp_audio.name
            audio_stream.download(filename=audio_path)

        st.success("Audio downloaded successfully!")

        st.info("Transcribing using OpenAI Whisper API...")
        with open(audio_path, "rb") as audio_file:
            transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
        transcript = transcript_response["text"]

        st.success("Transcription complete!")
        st.subheader("📜 Transcript Preview")
        st.write(transcript[:1000] + "...")  # Show partial text

        st.info("Generating questions using AI...")
        qa_prompt = f"""
        Extract 5 multiple choice questions from this transcript.
        Provide 4 options each and clearly mark the correct answer.
        Use this format:

        Q: Question text?
        a) Option A
        b) Option B
        c) Option C
        d) Option D
        Answer: b) Option B
        Explanation: ...

        Transcript:
        {transcript}
        """

        qa_generator = pipeline("text-generation", model="gpt2", max_length=1024)
        quiz_output = qa_generator(qa_prompt, num_return_sequences=1)[0]["generated_text"]

        st.subheader("📝 Generated Quiz")
        for block in quiz_output.split("Q:")[1:]:
            question_parts = block.strip().split("Answer:")
            question = "Q: " + question_parts[0].strip()
            answer_block = question_parts[1].strip().split("Explanation:")
            answer = "Answer: " + answer_block[0].strip()
            explanation = "Explanation:" + answer_block[1].strip() if len(answer_block) > 1 else ""

            with st.expander(question):
                st.write(answer)
                if explanation:
                    st.markdown(f"**Explanation:** {explanation}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
