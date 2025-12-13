"""Utility script for building a Notion-powered RAG pipeline with Gemini."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from notion_client import Client as NotionClient

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "notion_documents"
DEFAULT_CHAT_MODEL = "gemini-3-pro-preview"
CHAT_MODEL_ENV = "GEMINI_CHAT_MODEL"

def ensure_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable '{name}' is required")
    return value

def get_persist_dir() -> Path:
    directory = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    return Path(directory)

def init_notion_client() -> NotionClient:
    notion_token = ensure_env_var("NOTION_API_KEY")
    return NotionClient(auth=notion_token)

def get_chat_model_name() -> str:
    configured = os.getenv(CHAT_MODEL_ENV)
    if not configured:
        return DEFAULT_CHAT_MODEL
    return configured if configured.startswith("models/") else f"models/{configured}"

def extract_title(page: Dict) -> str:
    properties = page.get("properties", {})
    for prop in properties.values():
        if prop.get("type") == "title":
            return "".join(part.get("plain_text", "") for part in prop.get("title", []))
    return page.get("id", "")

def extract_rich_text(rich_text: Sequence[Dict]) -> str:
    return "".join(part.get("plain_text", "") for part in rich_text)

TEXT_BLOCK_KEYS = {
    "paragraph",
    "heading_1",
    "heading_2",
    "heading_3",
    "bulleted_list_item",
    "numbered_list_item",
    "quote",
    "callout",
    "to_do",
    "toggle",
}

def block_to_text(block: Dict) -> str:
    block_type = block.get("type")
    if block_type not in TEXT_BLOCK_KEYS:
        return ""
    value = block.get(block_type, {})
    text = extract_rich_text(value.get("rich_text", []))
    if block_type == "to_do":
        prefix = "[x]" if value.get("checked") else "[ ]"
        return f"{prefix} {text}".strip()
    return text.strip()

def fetch_page_content(client: NotionClient, page_id: str) -> List[str]:
    body: List[str] = []
    cursor = None
    while True:
        response = client.blocks.children.list(block_id=page_id, start_cursor=cursor)
        for block in response.get("results", []):
            text = block_to_text(block)
            if text:
                body.append(text)
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return body

def build_document_from_page(client: NotionClient, page_id: str) -> Optional[Document]:
    page = client.pages.retrieve(page_id=page_id)
    title = extract_title(page)
    content_lines = fetch_page_content(client, page_id)
    parts = [part for part in [title, "\n".join(content_lines)] if part]
    if not parts:
        return None
    page_text = "\n\n".join(parts)
    metadata = {
        "title": title or "Untitled",
        "notion_page_id": page_id,
        "url": page.get("url"),
    }
    return Document(page_content=page_text, metadata=metadata)

def chunk_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(list(documents))

def persist_documents(documents: List[Document], persist_dir: Path) -> None:
    ensure_env_var("GOOGLE_API_KEY")
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()

def load_vectorstore(persist_dir: Path) -> Chroma:
    ensure_env_var("GOOGLE_API_KEY")
    if not persist_dir.exists():
        raise RuntimeError(f"Persist directory '{persist_dir}' does not exist. Run the ingest step first.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

def ingest(page_id: str, reset: bool) -> None:
    persist_dir = get_persist_dir()
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    notion_client = init_notion_client()
    document = build_document_from_page(notion_client, page_id)
    if not document:
        print("No content was retrieved from the Notion page. Nothing to ingest.")
        return
    chunks = chunk_documents([document])
    persist_documents(chunks, persist_dir)
    title = document.metadata.get("title", "Untitled")
    print(f"Ingested Notion page '{title}' -> {len(chunks)} chunks into {persist_dir}.")

def answer(question: str) -> None:
    persist_dir = get_persist_dir()
    vectorstore = load_vectorstore(persist_dir)
    llm = ChatGoogleGenerativeAI(model=get_chat_model_name())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    documents = retriever.invoke(question)
    if not documents:
        print("No matching context was found in the vector store.")
        return
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_template(
        (
            "与えられた Notion ページの情報だけを使って日本語で回答してください。"
            "不明な場合はその旨を伝えてください。\n\n"
            "コンテキスト:\n{context}\n\n質問: {question}"
        )
    )
    chain = prompt | llm | StrOutputParser()
    answer_text = chain.invoke({"context": context, "question": question})
    print("\nAnswer:\n", answer_text)
    print("\nSources:")
    for doc in documents:
        title = doc.metadata.get("title", "Untitled")
        url = doc.metadata.get("url")
        ref = url or doc.metadata.get("notion_page_id")
        print(f"- {title} ({ref})")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini + Notion RAG helper")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Fetch a Notion page and rebuild the vector store")
    ingest_parser.add_argument(
        "--page-id",
        dest="page_id",
        help="Override NOTION_PAGE_ID environment variable",
    )
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector store before ingesting",
    )

    ask_parser = sub.add_parser("ask", help="Ask a question using the existing vector store")
    ask_parser.add_argument("question", help="質問内容")

    return parser

def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        page_id = args.page_id or ensure_env_var("NOTION_PAGE_ID")
        ingest(page_id=page_id, reset=args.reset)
    elif args.command == "ask":
        ensure_env_var("GOOGLE_API_KEY")
        answer(args.question)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
