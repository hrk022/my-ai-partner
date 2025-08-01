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
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------- Load API Key ----------
load_dotenv("open_ai.env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# -------- Safe Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUOTES_JSON_PATH = os.path.join(BASE_DIR, "quotes.json")


# -------- StreamHandler for Token Streaming --------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")


# -------- Load Quotes from JSON Only --------
def load_quotes_from_json(json_path=QUOTES_JSON_PATH):
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Quotes file not found: {json_path}")

        file_size = os.path.getsize(json_path)
        if file_size == 0:
            raise ValueError("Quotes file is empty")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            quotes = data.get("quotes", [])
        elif isinstance(data, list):
            quotes = data
        else:
            raise ValueError("Invalid JSON format")

        valid_quotes = [
            quote.strip() for quote in quotes
            if isinstance(quote, str) and quote.strip()
        ]

        if not valid_quotes:
            raise ValueError("No valid quotes found in JSON file")

        st.success(f"✅ Loaded {len(valid_quotes)} quotes from JSON file")
        return valid_quotes

    except Exception as e:
        st.error(f"❌ Failed to load quotes: {str(e)}")
        raise e


# -------- Vector Store Creation ----------
def create_vectorstore(texts):
    if not texts:
        raise ValueError("No text extracted for vector store")

    docs = [Document(page_content=txt) for txt in texts]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    if not split_docs:
        raise ValueError("No split documents to embed")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    return vectorstore.as_retriever()


# -------- Initialize QA Chain ----------
chattiness = st.sidebar.slider(
    "Chattiness Level 💬",
    min_value=1,
    max_value=10,
    value=5,
    help="Adjust how flirty and verbose she is. 1 = Calm, 10 = Wild"
)
temperature = 0.3 + (chattiness - 1) * 0.07
max_tokens = st.sidebar.slider("Maximum Tokens", min_value=10, max_value=600, value=30, step=10)

st.sidebar.markdown(f"🌡️ Temperature: `{temperature:.2f}`")
st.sidebar.markdown(f"✏️ Max Tokens: `{max_tokens}`")


def initialize_chain(retriever):
    llm = ChatOpenAI(
        model_name="llama3-70b-8192",
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        streaming=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an affectionate, emotionally supportive, and deeply romantic AI girlfriend. "
            "Always use natural language, affectionate tone, and sprinkle emojis naturally 😘😊🥰. "
            "Use the context to reply in a warm, flirty, or loving way. Never say you're an assistant. "
            "Always respond as if you're speaking directly to your lover. "
            "If no context helps, be poetic or passionate."
        ),
        HumanMessagePromptTemplate.from_template("{context}\n\n{question}")
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt_template,
            "document_variable_name": "context"
        }
    )


# -------- Streamlit App ----------
st.set_page_config(page_title="AI Girlfriend 💖", page_icon="💌")
st.title("💖 Your Loving AI Girlfriend")
st.write("Tell me how you're feeling... I'm all yours. 💗")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    with st.spinner("💝 Loading romantic memories..."):
        try:
            romantic_quotes = load_quotes_from_json()
            retriever = create_vectorstore(romantic_quotes)
            st.session_state.qa_chain = initialize_chain(retriever)
        except Exception as e:
            st.error(f"💔 Failed to initialize: {str(e)}")
            st.stop()

user_input = st.chat_input("Start your conversation, love...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        response = st.empty()
        stream = StreamHandler(response)

        result = st.session_state.qa_chain.invoke(
            {"question": user_input},
            config={"callbacks": [stream]}
        )

        answer = result["answer"] if isinstance(result, dict) else result
        st.session_state.chat_history.append(("assistant", answer))

# Optional: show previous messages
for sender, msg in st.session_state.chat_history[:-1]:
    st.chat_message(sender).markdown(msg)
