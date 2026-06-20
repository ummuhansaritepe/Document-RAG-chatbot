import streamlit as st
from dotenv import load_dotenv

from rag import build_chunks, build_vector_store, generate_answer, search_similar_chunks


load_dotenv()

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("Document RAG Chatbot")
st.write("Upload PDF, TXT, CSV, or Excel files and ask questions about their content.")

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "csv", "xlsx"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Process documents"):
        with st.spinner("Reading documents and creating embeddings..."):
            chunks = build_chunks(uploaded_files)
            vector_index, stored_chunks = build_vector_store(chunks)

            st.session_state.vector_index = vector_index
            st.session_state.chunks = stored_chunks
            st.session_state.messages = []

        st.success(
            f"Processed {len(uploaded_files)} file(s) and created {len(chunks)} chunks."
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents")

if question:
    if st.session_state.vector_index is None or st.session_state.chunks is None:
        st.warning("Please upload and process documents first.")
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                relevant_chunks = search_similar_chunks(
                    question=question,
                    index=st.session_state.vector_index,
                    chunks=st.session_state.chunks,
                    top_k=5,
                )

                answer = generate_answer(question, relevant_chunks)

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )