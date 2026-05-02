import os

from gitdb.fun import chunk_size
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA

load_dotenv()


class RAGEngine:
    def __init__(self, docs_folder: str = "documents", persist_dir: str = "vectorstore"):
        self.docs_folder = Path(docs_folder)
        self.persist_dir = persist_dir

        # Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=50,
            length_function=len
        )

        # Embeddings & LLM
        self.embeddings = OllamaEmbeddings(model="llama3.1", base_url="http://localhost:11434")
        self.llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434", temperature=0.7)

        #self.embeddings = OpenAIEmbeddings()
        #self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

        # Vector Store laden oder erstellen
        self.vectorstore = self._load_or_create_vectorstore()

        # QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def _load_or_create_vectorstore(self):
        """Lädt bestehenden Vectorstore oder erstellt neuen aus Dokumenten"""
        if os.path.exists(self.persist_dir) and self._has_index():
            print("Lade bestehenden Vectorstore...")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            print("Erstelle neuen Vectorstore aus Dokumenten...")
            return self._create_vectorstore()

    def _has_index(self) -> bool:
        """Prüft ob Index-Dateien existieren"""
        index_path = Path(self.persist_dir) / "index"
        return index_path.exists()

    def _create_vectorstore(self) -> Chroma:
        """Lädt Dokumente und erstellt Vectorstore"""
        print("🔍 Schritt 1: Prüfe documents-Ordner...")
        if not self.docs_folder.exists():
            self.docs_folder.mkdir(parents=True)
            raise FileNotFoundError(f"Ordner {self.docs_folder} nicht gefunden.")
        print("📄 Schritt 2: Lade PDF-Dokumente...")

        # Alle PDFs laden
        loader = DirectoryLoader(
            self.docs_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        print(f"   ✅ {len(documents)} Dokumente geladen")

        if not documents:
            raise ValueError("Keine PDF-Dokumente gefunden.")

        # Text splitten
        print("✂️  Schritt 3: Splitte Text in Chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Erstelle {len(chunks)} Text-Chunks aus {len(documents)} Dokumenten.")

        # Vectorstore erstellen und persistieren
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print("   ✅ Vectorstore erstellt!")
        vectorstore.persist()
        print("💾 Vectorstore gespeichert!")

        return vectorstore

    def query(self, question: str) -> dict:
        """Stellt eine Frage und gibt Antwort + Quellen zurück"""
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unbekannt") for doc in result["source_documents"]]
        }

    def reset_vectorstore(self):
        """Löscht den Vectorstore für Neuerstellung"""
        import shutil
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        self.vectorstore = self._create_vectorstore()