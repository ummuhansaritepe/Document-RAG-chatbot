import os
from io import BytesIO

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "source": uploaded_file.name,
                    "location": f"page {page_number}",
                    "text": text,
                }
            )

    return pages


def extract_text_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8", errors="ignore")

    return [
        {
            "source": uploaded_file.name,
            "location": "text file",
            "text": text,
        }
    ]


def extract_text_from_csv(uploaded_file):
    dataframe = pd.read_csv(uploaded_file)
    text = dataframe.to_string(index=False)

    return [
        {
            "source": uploaded_file.name,
            "location": "csv file",
            "text": text,
        }
    ]


def extract_text_from_excel(uploaded_file):
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    documents = []

    for sheet_name, dataframe in sheets.items():
        text = dataframe.to_string(index=False)

        documents.append(
            {
                "source": uploaded_file.name,
                "location": f"sheet {sheet_name}",
                "text": text,
            }
        )

    return documents


def extract_documents_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    if file_name.endswith(".csv"):
        return extract_text_from_csv(uploaded_file)

    if file_name.endswith(".xlsx"):
        return extract_text_from_excel(uploaded_file)

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks(uploaded_files):
    all_chunks = []

    for uploaded_file in uploaded_files:
        documents = extract_documents_from_file(uploaded_file)

        for document in documents:
            chunks = chunk_text(document["text"])

            for chunk_index, chunk in enumerate(chunks, start=1):
                all_chunks.append(
                    {
                        "source": document["source"],
                        "location": document["location"],
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                )

    return all_chunks


def create_embeddings(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]

    return np.array(embeddings, dtype="float32")


def build_vector_store(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


def search_similar_chunks(question, index, chunks, top_k=5):
    question_embedding = create_embeddings([question])

    distances, indices = index.search(question_embedding, top_k)

    results = []

    for item_index in indices[0]:
        if item_index != -1:
            results.append(chunks[item_index])

    return results


def generate_answer(question, relevant_chunks):
    context_parts = []

    for chunk in relevant_chunks:
        context_parts.append(
            f"Source: {chunk['source']} | Location: {chunk['location']} | Chunk: {chunk['chunk_index']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You are a helpful document question-answering assistant.
Answer the user's question using only the provided document context.
If the answer is not in the context, say that you could not find the answer in the uploaded documents.
Always mention the source file and location when possible.
"""

    user_prompt = f"""
Question:
{question}

Document context:
{context}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content