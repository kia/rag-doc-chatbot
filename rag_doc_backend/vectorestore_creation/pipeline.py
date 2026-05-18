import logging
logger = logging.getLogger("uvicorn.error")
import time
from langchain_chroma import Chroma

class VectorstoreCreationPipeline:
    def __init__(self, has_index=None, collection_name=None, persist_dir=None, use_openai_api=False, ollama_ingest_batch_size=100, embeddings=None):
        self.has_index = has_index
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.use_openai_api = use_openai_api
        self.OLLAMA_INGEST_BATCH_SIZE = ollama_ingest_batch_size


    def create_vectorstore_pipeline(self, chunks: list, clear_existing: bool = False) -> Chroma:
        """Provider-aware ingestion pipeline for vectorstore creation."""
        if clear_existing and self.has_index:
            try:
                existing_store = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings,
                )
                existing_store.delete_collection()
            except Exception as e:
                logger.error(f"Failed to delete existing collection: {e}")
                pass

        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Dimension mismatch detected: {e}. Clearing vectorstore and retrying...")
                import shutil
                shutil.rmtree(self.persist_dir, ignore_errors=True)
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                vectorstore = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings,
                )
            else:
                raise e

        if self.use_openai_api:
            vectorstore.add_documents(chunks)
            return vectorstore

        total_chunks = len(chunks)
        batch_size = self.OLLAMA_INGEST_BATCH_SIZE
        logger.debug(f"🧱 Step 4: Ollama ingest pipeline in batches of {batch_size}...")
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

            remaining_batches = (total_chunks - end) / batch_size
            remaining_seconds = estimate_time * remaining_batches

            estimate_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))
            estimate_finish_time = time.strftime("%H:%M:%S", time.localtime(time.time()+ remaining_seconds))
            msg = f"   ✅ Indexed chunks {start + 1}-{end} / {total_chunks} | ⏱ Estimated time remaining: {estimate_time_str} | estimated finish time: {estimate_finish_time}"
            logger.debug(msg)

        return vectorstore
