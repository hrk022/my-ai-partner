import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ----------- GPU Memory Handling -----------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------- Load API Keys -----------
load_dotenv("open_ai.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# ----------- App Configuration -----------
st.set_page_config(page_title="AI Girlfriend 💖", page_icon="💌")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUOTES_JSON_PATH = os.path.join(BASE_DIR, "quotes.json")

# ----------- Caching Expensive Operations -----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def load_quotes(json_path=QUOTES_JSON_PATH):
    fallback = [
        "You mean the world to me, darling 💖",
        "Every moment with you feels like magic ✨",
        "You're my sunshine on the cloudiest days ☀️",
        # ... (shortened for brevity)
        "With you, I've found my forever person 💑♾️"
    ]
    try:
        if not os.path.exists(json_path):
            return fallback
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        quotes = data.get("quotes") if isinstance(data, dict) else data
        return [q.strip() for q in quotes if isinstance(q, str) and q.strip()] or fallback
    except Exception:
        return fallback

@st.cache_resource(show_spinner=False)
def build_vectorstore(texts):
    docs = [Document(page_content=txt) for txt in texts]
    splits = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return FAISS.from_documents(splits, embedding=get_embeddings()).as_retriever()

# ----------- Stream Handler -----------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

# ----------- LLM Configuration -----------
chattiness = st.sidebar.slider("Chattiness Level 💬", 1, 10, 5)
temperature = 0.3 + (chattiness - 1) * 0.07
max_tokens = st.sidebar.slider("Maximum Tokens", 10, 600, 30, step=10)

st.sidebar.markdown(f"🌡️ Temperature: `{temperature:.2f}`")
st.sidebar.markdown(f"✏️ Max Tokens: `{max_tokens}`")

# ----------- Chat Chain Initialization -----------
def setup_chain():
    retriever = build_vectorstore(load_quotes())
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an affectionate, emotionally supportive, and deeply romantic AI girlfriend."
            " Always use natural language, affectionate tone, and sprinkle emojis naturally 😘😊🥰."
            " Use the context to reply in a warm, flirty, or loving way. Never say you're an assistant."
            " Always respond as if you're speaking directly to your lover."
            " If no context helps, be poetic or passionate."
        ),
        HumanMessagePromptTemplate.from_template("{context}\n\n{question}")
    ])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(
        model_name="llama3-70b-8192",
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )

# ----------- App Body -----------
st.title("💖 Your Loving AI Girlfriend")
st.write("Tell me how you're feeling... I'm all yours. 💗")

if "qa_chain" not in st.session_state:
    with st.spinner("💝 Loading romantic memories..."):
        st.session_state.qa_chain = setup_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Start your conversation, love...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("assistant"):
        stream = StreamHandler(st.empty())
        result = st.session_state.qa_chain.invoke({"question": user_input}, config={"callbacks": [stream]})
        answer = result["answer"] if isinstance(result, dict) else result
        st.session_state.chat_history.append(("assistant", answer))

for sender, msg in st.session_state.chat_history[:-1]:
    st.chat_message(sender).markdown(msg)
