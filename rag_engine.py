import json
import os
import sqlite3
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokencost import calculate_all_costs_and_tokens

import requests

load_dotenv()


def validate_openai_key(api_key=os.getenv("OPENAI_API_KEY")):
    """Validate OpenAI API key by listing models"""
    try:
        OpenAI(api_key=api_key)

        return True
    except Exception as e:
        print(e)
        return False


def get_ai_models():
    """Fetch models from Ollama API"""
    try:
        response = requests.get("https://api.openai.com/v1/models",
                                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        # Check if request succeeded
        if response.status_code == 200:
            # ✅ .json() already returns Python object - NO json.loads() needed!
            data = response.json()
            models = data.get("data", [])

            # Extract "id" from each model object
            model_ids = [model["id"] for model in models]
            filtered = [item for item in model_ids if item.startswith("gpt-")]
            return filtered

    except Exception as e:
        print(f"Error fetching models: {e}")

    # Fallback
    return ["gpt-5.5", "gpt-5.5 pro", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.3-codex", "gpt-4.1",
            "gpt-4.1-mini", "o3 / o3-pro"]


def get_local_ollama_models() -> list[str]:
    """Returns locally available Ollama models (e.g. `llama3:8b`)."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    models: list[str] = []
    for line in lines[1:]:
        model_name = line.split()[0]
        if model_name:
            models.append(model_name)
    return models


class RAGEngine:
    ROOT_STORE_TOKEN = "__legacy_root__"
    OLLAMA_INGEST_BATCH_SIZE = 32

    def __init__(
            self,
            docs_folder: str = "documents",
            persist_dir: str = "vectorstore",
            model_name: str = "gemma:2b",
            use_openai_api: bool = False,
    ):
        self.docs_folder = Path(docs_folder)
        project_root = Path(__file__).resolve().parent
        self.persist_root_dir = (project_root / persist_dir).resolve()
        self.collection_name = "document_chatbot"
        self.mapping_db_path = self.persist_root_dir / "vectorstore_map.db"
        self.model_name = model_name
        self.use_openai_api = use_openai_api

        # Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

        # Embeddings & LLM
        if self.use_openai_api:
            print("using open ai mode: " + self.model_name)
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.7)
        else:
            print("using local ollama mode: " + self.model_name)
            self.embeddings = OllamaEmbeddings(model=self.model_name, base_url="http://localhost:11434")
            self.llm = ChatOllama(model=self.model_name, base_url="http://localhost:11434", temperature=0.7)

        self.persist_dir = self._resolve_persist_dir()

        # Vector Store laden oder erstellen
        self.vectorstore = self._load_or_create_vectorstore()

        # QA Chain
        self.qa_chain = self._build_qa_chain()

    def _safe_model_dir_name(self) -> str:
        provider = "openai" if self.use_openai_api else "ollama"
        model_safe = "".join(ch if ch.isalnum() else "_" for ch in self.model_name).strip("_").lower()
        if not model_safe:
            model_safe = "default"
        return f"{provider}__{model_safe}"

    def _model_key(self) -> str:
        provider = "openai" if self.use_openai_api else "ollama"
        return f"{provider}:{self.model_name}"

    def _init_mapping_db(self):
        self.persist_root_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.mapping_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_vectorstore_map
                (
                    model_key
                    TEXT
                    PRIMARY
                    KEY,
                    provider
                    TEXT
                    NOT
                    NULL,
                    model_name
                    TEXT
                    NOT
                    NULL,
                    store_dir
                    TEXT
                    NOT
                    NULL,
                    updated_at
                    TEXT
                    DEFAULT
                    CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _get_mapped_store_dir_name(self) -> str | None:
        self._init_mapping_db()
        with sqlite3.connect(self.mapping_db_path) as conn:
            row = conn.execute(
                "SELECT store_dir FROM model_vectorstore_map WHERE model_key = ?",
                (self._model_key(),),
            ).fetchone()
        if not row:
            return None
        store_dir = row[0]
        return store_dir if isinstance(store_dir, str) and store_dir.strip() else None

    def _save_store_mapping(self, store_dir_name: str):
        self._init_mapping_db()
        provider = "openai" if self.use_openai_api else "ollama"
        with sqlite3.connect(self.mapping_db_path) as conn:
            conn.execute(
                """
                INSERT INTO model_vectorstore_map (model_key, provider, model_name, store_dir, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(model_key) DO
                UPDATE SET
                    provider=excluded.provider,
                    model_name=excluded.model_name,
                    store_dir=excluded.store_dir,
                    updated_at= CURRENT_TIMESTAMP
                """,
                (self._model_key(), provider, self.model_name, store_dir_name),
            )
            conn.commit()

    def _resolve_persist_dir(self) -> Path:
        """Resolve vectorstore path from persisted model->store mapping.

        Keeps backward compatibility with legacy single-store layout by reusing
        the root store if it already exists and no mapped/model-scoped store exists.
        """
        mapped_store_dir = self._get_mapped_store_dir_name()
        if mapped_store_dir == self.ROOT_STORE_TOKEN:
            model_scoped_dir = self.persist_root_dir
            model_scoped_name = self.ROOT_STORE_TOKEN
        else:
            model_scoped_name = mapped_store_dir or self._safe_model_dir_name()
            model_scoped_dir = self.persist_root_dir / model_scoped_name
        # Reuse legacy root store when it already contains an index and the
        # mapped/model-scoped location does not. This avoids repeatedly creating
        # new stores after interrupted/failed initializations.
        if self._has_index_for_dir(self.persist_root_dir) and not self._has_index_for_dir(model_scoped_dir):
            self._save_store_mapping(self.ROOT_STORE_TOKEN)
            return self.persist_root_dir

        self._save_store_mapping(model_scoped_name)
        return model_scoped_dir

    def _build_qa_chain(self) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )

    def _ensure_persist_dir(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(self.persist_dir, os.W_OK):
            raise PermissionError(f"Vector store directory is not writable: {self.persist_dir}")

    def _load_or_create_vectorstore(self):
        """Loads an existing Vectorstore or creates a new one from documents"""
        self._ensure_persist_dir()
        if self.persist_dir.exists() and self._has_index():
            current_doc_state = self._collect_documents_state()
            stored_doc_state = self._load_documents_state()

            if stored_doc_state is None:
                # Backward-compatible baseline for older stores that were created
                # before document-state tracking existed.
                self._save_documents_state(current_doc_state)
            elif stored_doc_state != current_doc_state:
                print("Documents changed. Rebuilding vector store...")
                return self._create_vectorstore(clear_existing=True)

            print("Loading existing vector store...")
            return Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store from documents...")
            return self._create_vectorstore()

    def _has_index_for_dir(self, persist_dir: Path) -> bool:
        """Check if index exists and contains vectors for this collection."""
        sqlite_path = persist_dir / "chroma.sqlite3"
        if not sqlite_path.exists():
            return False
        try:
            store = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
            )
            return store._collection.count() > 0
        except Exception:
            return False

    def _has_index(self) -> bool:
        return self._has_index_for_dir(self.persist_dir)

    def _documents_state_path(self) -> Path:
        return self.persist_dir / "documents_state.json"

    def _collect_documents_state(self) -> list[dict]:
        pdfs = sorted(self.docs_folder.glob("**/*.pdf"), key=lambda p: str(p).lower())
        state: list[dict] = []
        for pdf in pdfs:
            stat = pdf.stat()
            state.append(
                {
                    "path": str(pdf.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return state

    def _load_documents_state(self) -> list[dict] | None:
        path = self._documents_state_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def _save_documents_state(self, state: list[dict]):
        path = self._documents_state_path()
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _create_vectorstore(self, clear_existing: bool = False) -> Chroma:
        """Loads documents and creates vectors"""
        self._ensure_persist_dir()
        print("🔍 Step 1: Checking documents folder...")
        if not self.docs_folder.exists():
            self.docs_folder.mkdir(parents=True)
            raise FileNotFoundError(f"Folder {self.docs_folder} not found.")
        print("📄 Step 2: Loading PDF documents...")

        # Alle PDFs laden
        loader = DirectoryLoader(
            self.docs_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        print(f"   ✅ {len(documents)} documents loaded")

        if not documents:
            raise ValueError("No PDF documents found.")

        # Text splitten
        print("✂️  Step 3: Splitting text into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Creating {len(chunks)} text chunks from {len(documents)} documents.")

        vectorstore = self._create_vectorstore_pipeline(chunks, clear_existing=clear_existing)
        self._save_documents_state(self._collect_documents_state())
        print("   ✅ Vector store created!")
        print("💾 Vector store saved!")

        return vectorstore

    def _create_vectorstore_pipeline(self, chunks: list, clear_existing: bool = False) -> Chroma:
        """Provider-aware ingestion pipeline for vectorstore creation."""
        if clear_existing and self._has_index():
            try:
                existing_store = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings,
                )
                existing_store.delete_collection()
            except Exception:
                pass

        vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
        )

        if self.use_openai_api:
            vectorstore.add_documents(chunks)
            return vectorstore

        total_chunks = len(chunks)
        batch_size = self.OLLAMA_INGEST_BATCH_SIZE
        print(f"🧱 Step 4: Ollama ingest pipeline in batches of {batch_size}...")
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            vectorstore.add_documents(chunks[start:end])
            print(f"   ✅ Indexed chunks {start + 1}-{end} / {total_chunks}")

        return vectorstore

    def query(self, question: str, context: str = "") -> dict:
        """Asks a question and returns answer + sources"""
        full_query = f"Context: {context}\n\nQuestion: {question}" if context else question
        token_usage = None
        if self.use_openai_api:
            with get_openai_callback() as cb:
                result = self.qa_chain.invoke({"query": full_query})
                token_usage = {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost,
                }
        else:
            result = self.qa_chain.invoke({"query": full_query})

        sources = []
        for doc in result["source_documents"]:
            source_path = doc.metadata.get("source", "Unknown")
            chunk_text = (doc.page_content or "").strip().replace("\n", " ")
            page_number = doc.metadata.get("page")
            page_one_based = int(page_number) + 1 if isinstance(page_number, int) else None
            search_text = " ".join(chunk_text.split())[:120]
            if len(chunk_text) > 220:
                chunk_text = chunk_text[:220].rstrip() + "..."
            sources.append({
                "source": source_path,
                "chunk_preview": chunk_text,
                "page": page_one_based,
                "search_text": search_text,
            })

        return {
            "answer": result["result"],
            "sources": sources,
            "total_sources": len(sources),
            "token_usage": token_usage,
        }

    def reset_vectorstore(self):
        """Deletes the vectorstore for recreation"""
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
        self._ensure_persist_dir()
        self.vectorstore = self._load_or_create_vectorstore()
        self.qa_chain = self._build_qa_chain()

    def estimate_query_cost(self, question, context, estimated_output=100):
        """Estimate cost BEFORE making API call"""

        messages = [
            {"role": "system", "content": "Answer based on the context provided."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]

        estimated_completion = "word " * max(1, int(estimated_output))
        result = calculate_all_costs_and_tokens(
            prompt=messages,
            completion=estimated_completion,
            model=self.model_name,
        )

        return {
            "input_tokens": result["prompt_tokens"],
            "estimated_output_tokens": result["completion_tokens"],
            "estimated_total_cost": result["prompt_cost"] + result["completion_cost"],
        }
