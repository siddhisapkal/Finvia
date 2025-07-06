import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import tempfile

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit UI setup
st.set_page_config(page_title="Finvia - AI Financial Advisor", layout="wide")
with st.container():
    st.markdown("""
        <style>
        body {
            background-color: #F5F7FA;
        }
        .title {
            font-size: 3em;
            color: #19456B;
            text-align: center;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #607D8B;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="title">💸 Finvia</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your path to smarter financial decisions 🚀</div>', unsafe_allow_html=True)

DOCS_FOLDER = "data/docs"

@st.cache_resource
def load_default_docs(folder_path):
    all_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file_name))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
            docs = splitter.split_documents(pages)
            all_docs.extend(docs)
    return all_docs

# Load default documents
if not os.path.exists(DOCS_FOLDER):
    st.error("❌ 'docs/' folder not found. Please add PDF files inside `data/docs/`.")
    st.stop()

default_docs = load_default_docs(DOCS_FOLDER)

# File uploader
st.markdown("### 📤 Upload your financial PDFs")
uploaded_files = st.file_uploader("E.g., Salary Slips, ITRs", type="pdf", accept_multiple_files=True)

user_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
            docs = splitter.split_documents(pages)
            user_docs.extend(docs)

# Combine all documents
all_docs = default_docs + user_docs
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_docs, embeddings)

# Gemini model setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True,
)

# Personalization inputs
st.markdown("---")
st.markdown("### 🔒 Personalize Your Advice")
personalize = st.radio("Do you want personalized answers?", ["No", "Yes"], index=0, horizontal=True)
user_profile = {}
if personalize == "Yes":
    col1, col2 = st.columns(2)
    with col1:
        user_profile['age'] = st.number_input("📅 Age", min_value=18, max_value=100, value=25)
        user_profile['goal'] = st.text_input("🎯 Financial Goal", placeholder="e.g., Buy a house")
    with col2:
        user_profile['income'] = st.number_input("💰 Monthly Income (₹)", value=50000)
        user_profile['risk'] = st.selectbox("📉 Risk Appetite", ["Low", "Medium", "High"])

# Chat interface
st.markdown("---")
st.markdown("### 💬 Ask Finvia About Your Finances")
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat history
for role, msg in st.session_state.chat:
    st.chat_message(role).markdown(msg)

user_input = st.chat_input("Ask anything like: How can I save tax this year?")
if user_input:
    st.session_state.chat.append(("user", user_input))

    # Retrieve relevant context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(user_input)  # Updated per deprecation
    pdf_context = "\n\n".join([doc.page_content for doc in docs])

    # Prepare profile
    if personalize == "Yes":
        profile_txt = f"""
        Age: {user_profile['age']}
        Monthly Income: ₹{user_profile['income']}
        Goal: {user_profile['goal']}
        Risk Appetite: {user_profile['risk']}
        """
    else:
        profile_txt = "None provided."

    # Final prompt
    final_prompt = f"""
You are **Finvia**, a friendly, smart financial advisor powered by AI.

Help the user make informed financial decisions based on:
- 📄 The document excerpts below
- 👤 Their personal financial profile (if given)

📄 Document Context:
{pdf_context}

👤 User Profile:
{profile_txt}

❓ User Question:
{user_input}

Instructions:
- Be friendly and supportive (not robotic)
- Keep advice simple and easy to follow
- Personalize suggestions where applicable
- Give 2–3 useful, actionable ideas
- Suggest things like SIPs, PPF, FDs, or tax-saving instruments
- Don’t say “ask a human advisor” — you *are* their assistant
- Speak with clarity, warmth, and assurance like a trusted financial guide

Start your reply warmly. End by asking if they'd like help with anything else.
"""
if user_docs:
    st.markdown("### 📑 Extracted Text from Your Uploaded PDFs")
    for i, doc in enumerate(user_docs[:3]):
        st.expander(f"Document {i+1} Preview").markdown(doc.page_content[:1000])

    try:
        response = llm.invoke(final_prompt)
        reply = response.content.strip()
    except Exception as e:
        reply = f"⚠️ Gemini API Error: {e}"

    st.session_state.chat.append(("assistant", reply))
    st.rerun()
