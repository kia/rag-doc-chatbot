import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from generation.build_prompt import build_cost_estimation_prompt
from generation.generate_answer import generate_answer
from indexing.embeddings import create_embeddings
from indexing.load_docs import load_and_split_documents, collect_documents_state
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieval.search_vectors import build_vector_search_chain
from tokencost import calculate_all_costs_and_tokens
from vectorestore_creation.store_mapping import StoreMapping

load_dotenv()

class RAGEngine:
    OLLAMA_INGEST_BATCH_SIZE = 32

    def __init__(
            self,
            docs_folder: str = "documents",
            persist_dir: str = "vectorstore",
            model_name: str = "gemma:2b",
            use_openai_api: bool = False,
            status_callback: Callable[[str], None] | None = None,
    ):
        self.docs_folder = Path(docs_folder)
        project_root = Path(__file__).resolve().parent
        self.persist_root_dir = (project_root / persist_dir).resolve()
        self.collection_name = "document_chatbot"
        self.mapping_db_path = self.persist_root_dir / "vectorstore_map.db"
        self.model_name = model_name
        self.use_openai_api = use_openai_api
        self.status_callback = status_callback

        # Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

        # Embeddings & LLM
        if self.use_openai_api:
            print("using open ai mode: " + self.model_name)
            self.embeddings = create_embeddings(self.use_openai_api, self.model_name)
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.7)
        else:
            print("using local ollama mode: " + self.model_name)
            self.embeddings = create_embeddings(self.use_openai_api, self.model_name)
            self.llm = ChatOllama(model=self.model_name, base_url="http://localhost:11434", temperature=0.7)

        store_map = StoreMapping(self.use_openai_api, self.model_name)
        self.persist_dir = store_map.resolve_persist_dir()
        print("persist dir: " + str(self.persist_dir))
        # Vector Store laden oder erstellen
        self.vectorstore = self._load_or_create_vectorstore()

        # QA Chain
        self.qa_chain = self._build_qa_chain()


    def _build_qa_chain(self) -> RetrievalQA:
        return build_vector_search_chain(self.llm, self.vectorstore)

    def _ensure_persist_dir(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(self.persist_dir, os.W_OK):
            raise PermissionError(f"Vector store directory is not writable: {self.persist_dir}")

    def _load_or_create_vectorstore(self):
        """Loads an existing Vectorstore or creates a new one from documents"""
        self._ensure_persist_dir()
        print("loading vector store? " + str(self.persist_dir.exists() and self._has_index()))
        if self.persist_dir.exists() and self._has_index():
            current_doc_state = self._collect_documents_state()
            stored_doc_state = self._load_documents_state()

            if stored_doc_state is None:
                # Backward-compatible baseline for older stores that were created
                # before document-state tracking existed.
                self._save_documents_state(current_doc_state)
            elif stored_doc_state != current_doc_state:
                print("Documents changed. Rebuilding vector store...")
                self._set_status("🧱 Creating new vector store...")
                return self._create_vectorstore(clear_existing=True)

            print("Loading existing vector store...")
            self._set_status("📦 Loading existing vector store...")
            return Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store from documents...")
            self._set_status("🧱 Creating new vector store...")
            return self._create_vectorstore()

    def _set_status(self, message: str):
        if self.status_callback is not None:
            self.status_callback(message)

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
        return collect_documents_state(self.docs_folder)

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
        chunks = load_and_split_documents(self.docs_folder, self.text_splitter)

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
        estimate_time = None
        for start in range(0, total_chunks, batch_size):
            start_time = time.time()
            end = min(start + batch_size, total_chunks)
            vectorstore.add_documents(chunks[start:end])
            elapsed_time = time.time() - start_time
            if estimate_time is None:
                estimate_time = elapsed_time
            else:
                estimate_time = (estimate_time + elapsed_time) / 2

            estimate_time_str = time.strftime('%H:%M:%S', estimate_time * (total_chunks - end) / (end - start + 1))
            print(f"   ✅ Indexed chunks {start + 1}-{end} / {total_chunks}", f"⏱ Estimated time remaining: {estimate_time_str} seconds")

        return vectorstore

    def query(self, question: str, context: str = "") -> dict:
        """Asks a question and returns answer + sources"""
        return generate_answer(self.qa_chain, question, self.use_openai_api, context)

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
        messages = build_cost_estimation_prompt(question, context)

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
