import sqlite3
from pathlib import Path


class StoreMapping:
    """Class for managing and storing mappings between different models and their configurations."""
    ROOT_STORE_TOKEN = "__legacy_root__"
    def __init__ (self,
        use_openai_api: bool = False,
        model_name: str = None
        ):
        self.use_openai_api = use_openai_api
        self.model_name = model_name
        self.persist_root_dir = Path.home() / ".rag_engine"
        self.mapping_db_path = self.persist_root_dir / "mapping.db"

    def resolve_persist_dir(self) -> Path:
        """Resolve vectorstore path from persisted model->store mapping.

        Uses a model-scoped subdirectory under the vectorstore root.
        Legacy `__legacy_root__` mappings are normalized to model-scoped paths.
        """
        mapped_store_dir = self._get_mapped_store_dir_name()
        model_scoped_name: str
        if isinstance(mapped_store_dir, str) and mapped_store_dir != self.ROOT_STORE_TOKEN:
            model_scoped_name = str(mapped_store_dir)
        else:
            model_scoped_name = self._safe_model_dir_name()
        model_scoped_dir = self.persist_root_dir / model_scoped_name

        self._save_store_mapping(model_scoped_name)
        return model_scoped_dir

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


    def _has_any_store_mappings(self) -> bool:
        self._init_mapping_db()
        with sqlite3.connect(self.mapping_db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM model_vectorstore_map").fetchone()
        return bool(row and row[0] > 0)


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





