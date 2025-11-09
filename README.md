# 📊 Ollama RAG 10-K Analyzer

A local Retrieval-Augmented Generation (RAG) system that reads a company's **10-K filing** (e.g., Tesla) and answers questions using **Llama 3 via Ollama** — fully offline on your Mac.

---

## 🧩 Features
- Extracts text from any 10-K PDF
- Splits text into overlapping chunks (800 / 100)
- Embeds with `all-MiniLM-L6-v2`
- Stores in FAISS vector DB
- Retrieves relevant context for each question
- Generates answers with `ChatOllama` (Llama3)
- Works **100% locally**, no API key needed

---

## ⚙️ Setup

```bash
git clone https://github.com/MayankKumarPokhriyal/ollama-rag-10k-analyzer.git
cd ollama-rag-10k-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
