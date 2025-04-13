import os
import streamlit as st
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
from pinecone import Pinecone, ServerlessSpec
import whisper
import yt_dlp
import uuid
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE"]["api_key"]
PINECONE_API_KEY = st.secrets["PINECONE"]["api_key"]
PINECONE_ENV = st.secrets["PINECONE"]["environment"]
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
INDEX_NAME = "youtube-assistant"

embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
whisper_model = whisper.load_model("base")
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(INDEX_NAME)

def mistral_response(prompt):
    api_url = f"https://api-inference.huggingface.co/models/{MISTRAL_MODEL}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(api_url, headers=headers, json=data)
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    return "⚠ LLM generation failed."

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([item['text'] for item in transcript])
        return text, 'en', 'subtitles'
    except:
        try:
            auto_text, auto_lang = try_auto_transcript(video_id)
            translated = GoogleTranslator(source=auto_lang, target='en').translate(auto_text)
            return translated, auto_lang, 'translated-subtitles'
        except:
            whisper_data, lang = transcribe_audio(video_id)
            text = whisper_data["text"]
            translated = GoogleTranslator(source=lang, target='en').translate(text)
            return translated, lang, 'whisper-translated', text

def try_auto_transcript(video_id):
    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    for t in transcripts:
        if t.is_generated:
            try:
                fetched = t.fetch()
                lang = t.language_code
                text = " ".join([i['text'] for i in fetched])
                return text, lang
            except:
                continue
    raise RuntimeError("No usable auto-generated subtitles.")

def transcribe_audio(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    uid = str(uuid.uuid4())[:8]
    temp_file = f"temp_{uid}.%(ext)s"
    output = f"audio_{uid}.mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3'
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    downloaded = temp_file.replace('%(ext)s', 'mp3')
    os.rename(downloaded, output)
    result = whisper_model.transcribe(output, task="transcribe")
    lang = result.get("language") or detect(result["text"])
    os.remove(output)
    return {"text": result["text"]}, lang

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

def store_chunks(chunks, namespace):
    vectors = [
        {"id": f"{namespace}-{i}", "values": embedder.encode(chunk).tolist(), "metadata": {"text": chunk}}
        for i, chunk in enumerate(chunks)
    ]
    index.upsert(vectors=vectors, namespace=namespace)

def is_video_processed(namespace):
    try:
        res = index.query(vector=[0.0]*384, top_k=1, namespace=namespace)
        return len(res.get("matches", [])) > 0
    except:
        return False

def search_similar(query, namespace):
    query_vec = embedder.encode(query).tolist()
    results = index.query(vector=query_vec, top_k=5, include_metadata=True, namespace=namespace)
    return [match["metadata"]["text"] for match in results["matches"]]

def translate_to_english(text, lang):
    if lang == "en": return text
    return GoogleTranslator(source=lang, target='en').translate(text)

def translate_to_native(text, lang):
    if lang == "en": return text
    return GoogleTranslator(source='en', target=lang).translate(text)

# === Streamlit UI ===
st.set_page_config(page_title="YouTube Mistral Assistant", layout="wide")
st.title("🧠 YouTube LLM Assistant (Mistral + Whisper + Pinecone)")

video_url = st.text_input("📺 Paste YouTube video URL")
force_reprocess = st.checkbox("♻ Reprocess even if cached")

if video_url:
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript_raw = ""

        with st.spinner("🔍 Processing..."):
            if not force_reprocess and is_video_processed(video_id):
                st.info("✅ Already processed. Skipping extraction.")
            else:
                result = get_transcript(video_id)
                if len(result) == 4:
                    transcript_en, lang, source, transcript_raw = result
                else:
                    transcript_en, lang, source = result
                    transcript_raw = transcript_en

                chunks = chunk_text(transcript_en)
                store_chunks(chunks, video_id)
                st.success("📥 Transcript stored in Pinecone.")

                prompt = f"Summarize the following:\n{transcript_en[:2000]}"
                summary_en = mistral_response(prompt)
                summary_native = translate_to_native(summary_en, lang)

                st.subheader("📝 Summary")
                st.markdown(f"🔤 English:** {summary_en}")
                if lang != 'en':
                    st.markdown(f"🌍 {lang.upper()}:** {summary_native}")

        if transcript_raw:
            st.markdown("### 📜 Full Transcript")
            st.text_area("Transcript", transcript_raw, height=300)

    except Exception as e:
        st.error(f"🚫 Error: {str(e)}")

# === Q&A ===
question = st.text_input("❓ Ask your question:")
if st.button("🧠 Answer"):
    try:
        lang_q = detect(question)
        question_en = translate_to_english(question, lang_q)
        video_id = video_url.split("v=")[-1].split("&")[0]
        top_chunks = search_similar(question_en, video_id)
        context = " ".join(top_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question_en}\nAnswer:"
        answer_en = mistral_response(prompt)
        answer_native = translate_to_native(answer_en, lang_q)

        st.markdown("### 🤖 Answer")
        st.markdown(f"🔤 English: {answer_en}")
        if lang_q != "en":
            st.markdown(f"🌍 {lang_q.upper()}:** {answer_native}")
    except Exception as e:
        st.error(f"⚠ {str(e)}")


