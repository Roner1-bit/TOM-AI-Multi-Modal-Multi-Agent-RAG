# TOM AI Workshop Snippets (Runnable Mini-Demos)

These snippets let you test each component separately (2–5 minutes each), then assemble them.

## Prerequisites (macOS)
- Python 3.9+ (your `mm-agent` env)

### System deps
```bash
brew install ffmpeg portaudio
```

## Python deps (minimal set)
```bash
pip install --upgrade pip
pip install streamlit sounddevice soundfile numpy openai-whisper pillow python-dotenv
pip install langchain langchain-community langchain-core langchain-groq langchain-huggingface groq
pip install pypdf faiss-cpu sentence-transformers
```

## Recommended Run Order (Workshop Flow)
1. **Mic hardware**: `01_mic_test.py`  → creates `mic_test.wav`
2. **Whisper transcription**: `02_whisper_transcribe.py`  → transcribes `mic_test.wav`
3. **Streamlit Start/Stop (voice)**: `03_streamlit_voice.py`  → creates `streamlit_voice.wav`
4. **RAG retrieval only (no LLM)**: `04_rag_retrieval_only.py`  → requires `sample.pdf` or `sample.txt`
5. **RAG answer with Groq**: `05_rag_answer_groq.py`  → requires `GROQ_API_KEY`
6. **Router demo**: `06_router_demo.py`
7. **Mini Streamlit RAG (no LLM)**: `07_streamlit_rag_mini.py`
8. **Regular QA (Groq)**: `08_language_qa_groq.py`
9. **Chat memory demo**: `09_language_chat_memory.py`
10. **Code helper**: `10_code_helper_groq.py`
11. **Vision assistant (project)**:
    - CLI: `11A_use_project_vision_assistant.py`
    - Streamlit: `11B_streamlit_vision_mini.py`
12. **Router end-to-end**: `12_router_end_to_end.py`
13. **Mini TOM (text router)**: `13_streamlit_router_text_only.py`
14. **Mini TOM (all agents, no voice)**: `14_streamlit_all_agents_mini.py`

## Environment variables
For Groq-based snippets:
create .env file in the project root.
add this: GROQ_API_KEY="your_key_here"
```

## Notes / Troubleshooting
### macOS mic permission
Enable the app you run from:
- System Settings → Privacy & Security → Microphone → enable Terminal / VS Code, etc.
Then fully quit & reopen that app.

### Sample documents
Place `sample.pdf` or `sample.txt` in the same folder as the scripts for RAG demos.

### Vision demos
`11A/11B/12/14` place sample.png image in the same folder as the scripts for visual demos

### Running sample
From this folder:
```bash
python3 01_mic_test.py
streamlit run 03_streamlit_voice.py
```

Happy workshoping!
