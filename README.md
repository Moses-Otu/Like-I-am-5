# Nigeria Legal AI Assistant

**A Telegram-based Retrieval-Augmented Generation (RAG) assistant** that helps users query Nigerian legal documents (Constitution, Tax Acts, Labour laws) using OpenAI for LLM responses and Pinecone for vector search.

---

## Features

- PDF parsing (using `pdfplumber`) to extract legal text.
- Document chunking and embeddings with OpenAI `text-embedding-3-large`.
- Vector search and storage using Pinecone.
- Conversational answers via OpenAI chat models.
- Telegram bot interface for real-time user interaction.
- `.env` support for storing API keys locally.

---

## Tech stack

- Python 3.9+
- `pdfplumber` — PDF extraction
- `openai` — embeddings & chat completions
- `pinecone` — vector database
- `python-telegram-bot` — Telegram integration
- `python-dotenv` — load `.env` variables
- `tqdm` — progress bars

---

## Quickstart

### 1. Clone repo
```bash
git clone https://github.com/Moses-Otu/Like-I-am-5.git
cd Like-I-am-5
