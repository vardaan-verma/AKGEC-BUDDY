# 🤖 AKGEC Buddy — Smart India Hackathon Project by Team Intellivolve


<p align="center">
  <img src="https://img.shields.io/badge/Project-AKGEC%20Buddy-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Built%20For-Smart%20India%20Hackathon-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-yellow?style=for-the-badge&logo=javascript" />
  <img src="https://img.shields.io/badge/AI-NLP%20%26%20RAG-blue?style=for-the-badge&logo=openai" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>


### 🌐 A Multilingual, Context-Aware Chatbot with Self-Learning Capabilities

**AKGEC Buddy** is an intelligent, multilingual, and knowledge-driven chatbot designed for **Ajay Kumar Garg Engineering College (AKGEC)** as part of the **Smart India Hackathon (SIH)**.  
It can understand natural language queries through **intent detection**, fetch accurate responses from a **knowledge base**, and even **learn automatically** from official college sources using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Project Overview

### 🧩 Problem Statement  
> To develop a **multilingual agnostic chatbot** that understands user intent, answers queries from an internal knowledge base, and forwards unresolved queries to relevant staff members.

### 🎯 Solution — AKGEC Buddy
- 💬 **Intent Detection**: Understands user questions across multiple languages.  
- 📚 **Knowledge Base Integration**: Fetches accurate and context-aware answers.  
- 🌍 **Multilingual Support**: Communicates in English, Hindi, and more.  
- 🧠 **Self-Learning (RAG)**: Periodically scans the official college website to learn new updates (like holidays, fee deadlines, or notices) and automatically updates its knowledge base.  
- 🧾 **Escalation System**: Forwards unresolved queries to staff for manual handling.  
- ⚡ **FastAPI Backend** with modular architecture for scalability.

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | FastAPI (Python) |
| **AI/ML** | NLP + Intent Detection + RAG |
| **Database** | MongoDB |
| **Other Tools** | LangChain, OpenAI API, BeautifulSoup, Python-dotenv |

---

## 📂 Project Structure

chatbot/
├── backend/
│ ├── main.py
│ ├── routes/
│ ├── models/
│ ├── utils/
│ └── knowledge_base/
├── frontend/
│ ├── index.html
│ ├── style.css
│ └── script.js
├── .env # ⚠️ Contains API keys (do NOT upload)
├── requirements.txt
├── .gitignore
└── README.md


## ⚙️ Setup Instructions

1️⃣ Clone the Repository
''bash
git clone https://github.com/<your-username>/chatbot.git
cd chatbot

2️⃣ Create a Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Setup Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_api_key_here
MONGODB_URI=your_mongodb_connection_string

5️⃣ Run the Server
uvicorn backend.main:app --reload


Then open your browser at http://127.0.0.1:8000

🧠 How It Learns

AKGEC Buddy uses a RAG (Retrieval-Augmented Generation) pipeline:

Scrapes official AKGEC web pages for new updates.

Converts useful text data into embeddings.

Stores embeddings in a vector database.

When users ask something, it retrieves the most relevant chunks for context.

Updates the knowledge base automatically with new verified data.

This allows continuous, self-improving responses without manual intervention.


📜 License

This project is licensed under the MIT License — feel free to use and adapt it with proper credit.

💬 Connect

https://github.com/vardaan-verma

https://in.linkedin.com/in/vardaan-verma-
