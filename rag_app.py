import os
import shutil
import tempfile
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.storage import InMemoryByteStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import fitz
import re

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat — AI Testing Paper",
    page_icon="📄",
    layout="wide"
)

# ── Helper functions ──────────────────────────────────────────
def load_and_clean(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    full_text = re.sub(r'\n\d+', '', full_text)                            # remove page Number headers
    full_text = re.sub(r'\nTesting Artiﬁcial Intelligence', '', full_text) # remove repeating header
    full_text = re.sub(r'G. Numan\n', '', full_text)                       # remove repeating header          
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)                       # collapse blank lines
    full_text = re.sub(r'-\r?\n', '-', full_text)                          # fix hyphenated line breaks
    full_text = full_text.strip()
    return full_text.strip()

def build_retriever(clean_text):
    source_doc = Document(
        page_content=clean_text,
        metadata={"source": "Testing_Artificial_Intelligence.pdf"}
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # use /tmp so it works on Streamlit Cloud
    db_path = os.path.join(tempfile.gettempdir(), "rag_parent_child_db")

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    child_vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embedding_model,
        persist_directory=db_path
    )

    parent_store = InMemoryByteStore()

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=parent_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    retriever.add_documents([source_doc])

    return retriever, child_vectorstore._collection.count()

def build_chain(retriever):
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an assistant answering questions about an AI testing research paper.
Use ONLY the context below to answer. If the answer is not in the context, say "I don't have enough information to answer this."
Be concise and specific. Answer in 2-4 sentences maximum.
Do not include information that is not directly relevant to the question.

Context:
{context}

Question: {question}

Answer:"""
    )

    def get_answer(question):
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        sources = [
            doc.page_content[:120] + "..."
            for doc in docs
        ]
        return answer, sources

    return get_answer

# ── Load everything once using session state ──────────────────
@st.cache_resource(show_spinner="Loading and indexing document...")
def initialize():
    clean_text = load_and_clean("Testing_Artificial_Intelligence.pdf")
    retriever, chunk_count = build_retriever(clean_text)
    get_answer = build_chain(retriever)
    return get_answer, chunk_count

get_answer, chunk_count = initialize()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document info")
    st.info("**Testing Artificial Intelligence**\nGerard Numan")
    st.metric("Child chunks indexed", chunk_count)
    st.caption("Model: Llama 3.1 8b (Groq)")
    st.caption("Retriever: Parent-child")
    st.caption("Embeddings: all-MiniLM-L6-v2")

    st.divider()
    show_sources = st.toggle("Show sources", value=False)

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ── Chat area ─────────────────────────────────────────────────
st.title("💬 RAG Chat — AI Testing Paper")
st.caption("Ask anything about the paper. Answers are grounded in the document only.")

# initialise message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! Ask me anything about the AI Testing paper by Gerard Numan.",
         "sources": []}
    ]

# render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander("Sources used"):
                for i, src in enumerate(msg["sources"]):
                    st.caption(f"**Chunk {i+1}:** {src}")

# handle new input
if question := st.chat_input("Ask a question about the paper..."):

    # show user message immediately
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.write(question)

    # generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(question)
        st.write(answer)
        if show_sources and sources:
            with st.expander("Sources used"):
                for i, src in enumerate(sources):
                    st.caption(f"**Chunk {i+1}:** {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )