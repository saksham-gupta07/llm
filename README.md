LLM-Powered YouTube Assistant


Project Overview:

This project is a YouTube LLM-powered assistant built to process YouTube videos by extracting transcripts via multiple methods (YouTube’s transcript API, auto-generated subtitles, or Whisper transcription), storing them in a Pinecone vector database, and then allowing users to interact with the content via a Q&A interface—all powered by a Mistral-based LLM from Hugging Face.

Key Components:

Transcript Extraction:
Uses the youtube-transcript-api to fetch subtitles. If available, auto-generated transcripts are used; otherwise, Whisper is employed for transcription. Results are translated to English using Deep Translator if necessary.

Text Processing & Embedding:
The transcript is chunked using LangChain’s RecursiveCharacterTextSplitter and converted to embeddings using SentenceTransformer (paraphrase-MiniLM-L6-v2). These embeddings are upserted into a Pinecone index for efficient retrieval.

LLM Generation:
A call is made to the Hugging Face Inference API to generate responses from the Mistral model (mistralai/Mistral-7B-Instruct-v0.1) based on a prompt combining the retrieved context and a user’s query.

UI & Interaction:
The UI is built using Streamlit, enabling a simple interface where users can:
Paste a YouTube video URL
Process and store the transcript (if not already cached)
Ask questions about the video content and receive both English and translated answers

Tech Stack:

Streamlit: For building a rapid, interactive UI
Hugging Face Transformers: For LLM inference (Mistral model)
LangChain: For text splitting and prompt construction
SentenceTransformer: For generating vector embeddings
Pinecone: For storing embeddings and performing efficient vector search
YouTube Transcript API: For fetching video transcripts
Whisper (OpenAI): For audio transcription when subtitles aren’t available
Deep Translator: For translation services
yt-dlp: For downloading YouTube audio
Python Libraries: Requests, uuid, warnings, etc.

Setup & Installation:

Prerequisites
Python 3.11 (or as specified)
A Pinecone account and an API key
A Hugging Face account with a valid API token (ensure your model license is accepted if required)


Project Structure

llm/
├── llm.py               # Main application code
├── requirements.txt     # All required packages with versions
├── README.md            # This file


Transcript Extraction:
Checks if a transcript is available via YouTube Transcript API.
Falls back to auto-generated transcripts if necessary.
Uses Whisper for transcription if transcripts can’t be fetched.
Utilizes Google’s Deep Translator for translating to English, if needed.

Data Processing & Storage:
The transcript is chunked (approximately 500 characters per chunk) using LangChain’s splitter.
Each chunk is embedded via the SentenceTransformer model.
Embeddings are upserted into a Pinecone index, using the video ID as the namespace.

Question Answering:
A user’s question is translated to English if necessary.
The query is embedded and used to retrieve top matching transcript chunks from Pinecone.
A prompt is generated combining the retrieved context and the user’s question.
The prompt is sent to the Mistral model via Hugging Face’s Inference API to generate an answer.
The response is then translated back to the user’s language if required.

UI/UX:
The interface guides users through video URL input, transcript processing, and Q&A.
Feedback is provided through Streamlit’s spinners, info messages, and error messages to ensure clarity.

Troubleshooting:

Torch / File Watcher Warnings:
Some warnings related to Torch’s internal file-watching mechanism may appear but do not affect functionality.

Dependency Conflicts:
If you run into conflicts with libraries such as NumPy or urllib3 adjust to versions in requirements.txt.

API Keys & Tokens:
Ensure that you have valid Pinecone and Hugging Face API keys and that they are correctly set in your code.

Conclusion:
This solution demonstrates an innovative approach to processing YouTube content with LLM-powered insights, leveraging state-of-the-art tools like Mistral, Whisper, and Pinecone to deliver an interactive assistant via a clean, modern UI.
Your creativity in integrating these diverse tools under a single workflow showcases the potential of LLM-powered solutions for real-world problems.