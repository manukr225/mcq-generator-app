import streamlit as st
import tempfile
import openai
import os
from yt_dlp import YoutubeDL
from urllib.parse import urlparse, parse_qs

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to clean and standardize YouTube links
def clean_youtube_url(url):
    try:
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
            return f"https://www.youtube.com/watch?v={video_id}"
        elif "watch" in url:
            parsed = urlparse(url)
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            return f"https://www.youtube.com/watch?v={video_id}"
        return url
    except:
        return url

# Streamlit UI
st.title("🎓 YouTube Video to MCQ Quiz Generator")
yt_link = st.text_input("Paste YouTube video link")

if yt_link:
    yt_link = clean_youtube_url(yt_link)
    st.success(f"✅ Cleaned link: {yt_link}")

    try:
        st.info("📥 Downloading audio using yt_dlp...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_path = temp_audio.name

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_link])

        st.success("✅ Audio downloaded successfully!")

        st.info("🔊 Transcribing using OpenAI Whisper API...")
        with open(audio_path, "rb") as audio_file:
            transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
        transcript = transcript_response["text"]

        st.success("📝 Transcription complete!")
        st.subheader("📜 Transcript Preview")
        st.write(transcript[:1000] + "...")

        st.info("🤖 Generating MCQ quiz using GPT-3.5...")

        qa_prompt = f"""
        Generate 5 multiple choice questions from the transcript below.
        Each question should have 4 options (a-d), and clearly mark the correct answer and explanation.

        Format:
        Q: <Question>
        a) Option A
        b) Option B
        c) Option C
        d) Option D
        Answer: <correct option letter>) <Correct Option Text>
        Explanation: <Short explanation>

        Transcript:
        {transcript}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": qa_prompt}],
            temperature=0.7
        )

        quiz_output = response["choices"][0]["message"]["content"]

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
