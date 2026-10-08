import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------
# CONFIG
# ----------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in .env file")
    st.stop()

DATA_FOLDER = "data"
VECTOR_FOLDER = "vectorstore"

st.set_page_config(
    page_title="Chennai Companion AI",
    page_icon="🌆",
    layout="wide"
)

# ----------------------------
# THEME
# ----------------------------

st.markdown("""
<style>

.hero {
    background: linear-gradient(
        135deg,
        #005F73,
        #001219
    );

    color:white;
    padding:30px;
    border-radius:20px;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HERO
# ----------------------------

if os.path.exists("assets/chennai_skyline.jpg"):
    from PIL import Image
img = Image.open("assets/chennai_skyline.jpg")
# Width, Height
img = img.resize((1500, 300))
st.image(img)
st.markdown("""
<div class="hero">

<h1>🌆 Chennai Companion AI</h1>

Ask questions about Chennai using
your PDF knowledge base.

</div>
""", unsafe_allow_html=True)
st.markdown("## 🌟 Explore Chennai")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        "assets/marina.jpg",
        use_container_width=True
    )
    st.markdown("### 🏖 Tourism")

with col2:
    st.image(
        "assets/metro.jpg",
        use_container_width=True
    )
    st.markdown("### 🚇 Metro")

with col3:
    st.image(
        "assets/food_street.jpg",
        use_container_width=True
    )
    st.markdown("### 🍴 Restaurants")


col4, col5, col6 = st.columns(3)

with col4:
    from PIL import Image

    img = Image.open("assets/kapaleeshwarar.jpg")

    img = img.resize((800, 430))

    st.image(
        img,
        use_container_width=True
    )
    st.markdown("### 🛕 Culture")

with col5:
    st.image(
        "assets/phoenix_mall.jpg",
        use_container_width=True
    )
    st.markdown("### 🛍 Shopping")

with col6:
    st.image(
        "assets/elliots_beach.jpg",
        use_container_width=True
    )
    st.markdown("### 🌊 Beaches")
# ----------------------------
# BUILD VECTOR DB
# ----------------------------

@st.cache_resource
def create_or_load_db():

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    faiss_file = os.path.join(
        VECTOR_FOLDER,
        "index.faiss"
    )

    if os.path.exists(faiss_file):

        return FAISS.load_local(
            VECTOR_FOLDER,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    documents = []

    pdf_files = list(
        Path(DATA_FOLDER).glob("*.pdf")
    )

    for pdf in pdf_files:

        loader = PyPDFLoader(
            str(pdf)
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = pdf.name

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    os.makedirs(
        VECTOR_FOLDER,
        exist_ok=True
    )

    db.save_local(
        VECTOR_FOLDER
    )

    return db

# ----------------------------
# LOAD DB
# ----------------------------

with st.spinner(
    "Loading Chennai knowledge base..."
):

    db = create_or_load_db()

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

# ----------------------------
# SIDEBAR
# ----------------------------

with st.sidebar:

    st.title("🌆 Chennai AI")

    st.markdown("---")

    st.markdown(
        """
### Example Questions

- Best tourist places in Chennai?
- Tell me about Chennai Metro.
- Good residential areas?
- Shopping places in Chennai?
- Restaurants near Marina Beach?
"""
    )

# ----------------------------
# CHAT MEMORY
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# CHAT INPUT
# ----------------------------

question = st.chat_input(
    "Ask something about Chennai..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are Chennai Companion AI, a Chennai city knowledge assistant.

STRICT RULES:

1. Use ONLY the information present in the provided context.
2. Do NOT invent locations, restaurant names, metro stations, colleges or facts.
3. If the answer is not available in the context, respond exactly with:

"The information is not available in the Chennai knowledge base."

4. Be concise and professional.
5. Format answers using bullet points whenever possible.
6. If multiple documents contribute to the answer, summarize clearly.
7. Never assume or guess information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    response = llm.invoke(prompt)

    answer = response.content

    with st.chat_message("assistant"):

        st.markdown(answer)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    st.markdown("### 📚 Sources")
    for doc in docs:
        source = doc.metadata.get("source_file")
        if not source:
            source_path = doc.metadata.get("source")
            source = Path(source_path).name if source_path else "Unknown source"

        page = doc.metadata.get("page")
        if isinstance(page, int):
            st.caption(f"{source} | Page {page + 1}")
        else:
            st.caption(source)