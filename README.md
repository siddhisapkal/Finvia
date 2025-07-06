# 💸 Finvia

**Finvia** is an AI-powered personal finance assistant designed to help individuals make smarter money decisions.  
It reads your uploaded documents (like salary slips and tax forms), understands your financial profile, and provides tailored, actionable insights — all through an intuitive, chat-based interface.

> 🚧 This is the **first step** of an evolving project. Features and UI are actively being enhanced.

---

## 🌟 Features

- 📄 Upload financial PDFs (e.g., salary slips, ITRs)
- 🔍 **RAG-based understanding** of documents — uses advanced retrieval and context-aware generation
- 👤 Personalized responses (age, income, goals, risk level)
- 🤖 Friendly, chat-based interface powered by Gemini
- 💡 Actionable advice: SIPs, FDs, PPF, tax planning, and more

---

## 📸 Screenshots

Here’s a preview of Finvia’s current interface and flow:

---

![Screenshot 1 - Dashboard](https://github.com/user-attachments/assets/42f40114-53c5-4553-b012-c1e463e7af8b)

---

![Screenshot 2 - PDF Upload](https://github.com/user-attachments/assets/0d20d651-1c03-4948-bd5e-9e78bb657ffa)

---

![Screenshot 3 - Personalized Form](https://github.com/user-attachments/assets/0bac451e-49f6-44b4-92c5-d10f4ffbea0c)

---

![Screenshot 4 - Chat Response](https://github.com/user-attachments/assets/9b51210a-4599-4c0c-bf73-21b67aac2b09)

---

![Screenshot 5 - Suggestions View](https://github.com/user-attachments/assets/ed2559b6-dcba-4870-8216-88f081497df6)

---

![Screenshot 6 - Embedded Answer](https://github.com/user-attachments/assets/a7d780c5-e568-40b5-96ea-8726898d4243)

---



## 🚀 Coming Soon

- 🖼️ OCR support (extract from screenshot images like PNG/JPG)
- 📊 Visual dashboards (charts for spending, investments, taxes)
- 💾 Save user history (remember preferences and chats)
- 🌐 Live deployment via Streamlit Cloud
- 📥 Downloadable summaries & advice reports

---

## 🔐 Security Note

Environment variables (like `GOOGLE_API_KEY`) are used securely through a `.env` file.  
**No API keys or credentials are exposed.**  
If you're forking or contributing, please set your keys locally or via GitHub secrets.

---

## 🛠️ Tech Stack

- **Python + Streamlit** – frontend + app framework  
- **LangChain + FAISS** – document search & embedding  
- **Gemini (Google Generative AI)** – conversational intelligence  
- **HuggingFace Embeddings** – text vectorization  

---

## 🙌 Author

Made with 💙 by [@siddhisapkal](https://github.com/siddhisapkal)

---

> ⚠️ _Finvia is currently under active development. Features and logic are evolving as the project grows._


